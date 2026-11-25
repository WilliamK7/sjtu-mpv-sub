// ==UserScript==
// @name         SJTU 课堂实录 mpv 双语流式播放
// @namespace    https://v.sjtu.edu.cn/
// @version      0.3.1
// @description  一键将交大课堂实录同一堂课分节合并为连续进度条流式唤起本地 mpv，并叠加阿里云百炼实时双语字幕
// @author       sjtu-mpv-sub
// @match        https://v.sjtu.edu.cn/jy-application-resourcemanage-ui/*
// @run-at       document-start
// @grant        GM_xmlhttpRequest
// @grant        GM_addStyle
// @grant        GM_getValue
// @grant        GM_setValue
// @grant        unsafeWindow
// @grant        window.close
// @connect      127.0.0.1
// @connect      localhost
// ==/UserScript==

(function () {
  'use strict';

  const win = typeof unsafeWindow !== 'undefined' ? unsafeWindow : window;
  const LOCAL_API = 'http://127.0.0.1:18732';
  const WAKE_URI = 'sjtu-mpv://wake';
  const API_PREFIX = '/jy-application-resourcemanage/';
  const GROUP_MAX_GAP_MS = 35 * 60 * 1000;
  const LOCAL_VIEW_STORE_KEY = 'sjtu_mpv_course_views';

  const state = {
    headers: {},
    teclId: null,
    courseName: '',
    currentCourseId: null,
    episodes: [],
    vodCache: new Map(),
    pptCache: new Map(),
    preferredView: 'ppt',
    viewManuallySelected: false,
    glossaryText: '',
    extraPrompt: '',
    forceRetranscribe: false,
    selectedIds: new Set(),
    showEpisodes: false,
    minimized: false,
    busy: false,
    allowWebVideoPlay: false,
    apiKeyConfigured: true,
    pollTimer: null,
    statusText: '就绪',
    statusType: 'info',
    progressPct: 0,
  };

  function readCourseViewMap() {
    try {
      if (typeof GM_getValue === 'function') {
        const stored = GM_getValue(LOCAL_VIEW_STORE_KEY, {});
        if (stored && typeof stored === 'object') return stored;
      }
    } catch {
      // Fallback below.
    }
    try {
      const raw = window.localStorage.getItem(LOCAL_VIEW_STORE_KEY);
      const parsed = raw ? JSON.parse(raw) : {};
      return parsed && typeof parsed === 'object' ? parsed : {};
    } catch {
      return {};
    }
  }

  function loadLocalCourseView(teclId) {
    if (!teclId) return null;
    const map = readCourseViewMap();
    const v = map[String(teclId)];
    return v === 'ppt' || v === 'camera' ? v : null;
  }

  function saveLocalCourseView(teclId, view) {
    if (!teclId || (view !== 'ppt' && view !== 'camera')) return;
    const map = Object.assign({}, readCourseViewMap(), { [String(teclId)]: view });
    try {
      if (typeof GM_setValue === 'function') {
        GM_setValue(LOCAL_VIEW_STORE_KEY, map);
      }
    } catch {
      // Ignore GM storage failure.
    }
    try {
      window.localStorage.setItem(LOCAL_VIEW_STORE_KEY, JSON.stringify(map));
    } catch {
      // Ignore localStorage failure.
    }
  }

  function setCurrentTeclId(rawTeclId) {
    const nextId = Number(rawTeclId || 0);
    if (!nextId) return;
    if (state.teclId !== nextId) {
      state.teclId = nextId;
      state.viewManuallySelected = false;
      state.preferredView = loadLocalCourseView(nextId) || 'ppt';
      scheduleUiSync();
    }
  }

  function isPlayCenter() {
    return (window.location.hash || '').includes('/play-center');
  }

  function pauseAllWebMedia() {
    state.allowWebVideoPlay = false;
    try {
      document.querySelectorAll('video, audio').forEach((el) => {
        try {
          el.autoplay = false;
          el.removeAttribute('autoplay');
          el.pause();
        } catch {
          // Ignore media element errors.
        }
      });
    } catch {
      // Ignore DOM not ready.
    }
  }

  function guardWebAutoplay(ev) {
    if (!isPlayCenter() || state.allowWebVideoPlay) return;
    const target = ev && ev.target;
    if (target && typeof target.pause === 'function') {
      try {
        target.autoplay = false;
        target.removeAttribute('autoplay');
        target.pause();
      } catch {
        // Ignore pause errors.
      }
    }
  }

  // Block web player auto-play on page load while preserving explicit clicks on the player.
  const origMediaPlay = win.HTMLMediaElement && win.HTMLMediaElement.prototype && win.HTMLMediaElement.prototype.play;
  if (origMediaPlay) {
    win.HTMLMediaElement.prototype.play = function () {
      if (isPlayCenter() && !state.allowWebVideoPlay) {
        try {
          this.autoplay = false;
          this.pause();
        } catch {
          // Ignore pause errors.
        }
        return Promise.resolve();
      }
      return origMediaPlay.apply(this, arguments);
    };
  }
  document.addEventListener('play', guardWebAutoplay, true);
  document.addEventListener('playing', guardWebAutoplay, true);
  document.addEventListener('pointerdown', (ev) => {
    if (!isPlayCenter()) return;
    const panel = document.getElementById('sjtu-mpv-panel');
    const modal = document.getElementById('sjtu-mpv-glossary-modal');
    if ((panel && panel.contains(ev.target)) || (modal && modal.contains(ev.target))) {
      return;
    }
    const target = ev.target;
    if (
      target &&
      typeof target.closest === 'function' &&
      target.closest('video, .vjs-tech, .xgplayer, [class*="player"], [class*="video"], [class*="play-btn"], [class*="control"]')
    ) {
      state.allowWebVideoPlay = true;
    }
  }, true);

  function parseDateTimeMs(str) {
    if (!str) return NaN;
    if (typeof str === 'number') return str;
    const normalized = String(str).trim().replace(' ', 'T');
    const ts = Date.parse(normalized);
    return Number.isFinite(ts) ? ts : NaN;
  }

  function formatDatePart(str) {
    if (!str) return '';
    return String(str).trim().slice(0, 10);
  }

  function formatPeriodLabel(ep) {
    const dt = formatDatePart(ep.courBeginTime);
    const num = ep.letiNumber != null ? `第${ep.letiNumber}节` : `ID ${ep.id}`;
    const timeRange = ep.courBeginTime && ep.courEndTime
      ? `${String(ep.courBeginTime).slice(11, 16)}-${String(ep.courEndTime).slice(11, 16)}`
      : '';
    return `${dt} ${num} ${timeRange}`.trim();
  }

  function captureHeader(name, value) {
    if (!name || value == null || value === '') return;
    const lower = String(name).toLowerCase();
    if (lower === 'authorization' || lower === 'jwt-token' || lower === 'version-info') {
      state.headers[name] = String(value);
    }
  }

  function extractHeadersFromInit(initHeaders) {
    if (!initHeaders) return;
    if (typeof Headers !== 'undefined' && initHeaders instanceof Headers) {
      initHeaders.forEach((v, k) => captureHeader(k, v));
      return;
    }
    if (Array.isArray(initHeaders)) {
      for (const pair of initHeaders) {
        if (Array.isArray(pair) && pair.length >= 2) {
          captureHeader(pair[0], pair[1]);
        }
      }
      return;
    }
    if (typeof initHeaders === 'object') {
      for (const [k, v] of Object.entries(initHeaders)) {
        captureHeader(k, v);
      }
    }
  }

  function parseQueryParam(urlStr, key) {
    try {
      const u = new URL(urlStr, window.location.origin);
      return u.searchParams.get(key);
    } catch {
      return null;
    }
  }

  function handleInterceptedJson(urlStr, json) {
    if (!json || typeof json !== 'object') return;
    const data = json.data;
    if (!data) return;

    if (urlStr.includes('/v1/subject_vod_list_new') && Array.isArray(data.records)) {
      const teclParam = parseQueryParam(urlStr, 'teclIds');
      if (teclParam) {
        setCurrentTeclId(teclParam);
      }
      state.episodes = data.records.slice();
      if (!state.courseName && data.records.length > 0) {
        state.courseName = data.records[0].subjName || data.records[0].teclName || '';
      }
      recomputeSelection(true);
      scheduleUiSync();
      return;
    }

    if (urlStr.includes('/v1/course_vod_urls_new') && data.id) {
      const cid = Number(data.id);
      if (state.currentCourseId !== cid && !state.busy) {
        pauseAllWebMedia();
      }
      state.vodCache.set(cid, data);
      state.currentCourseId = cid;
      if (data.teclId) setCurrentTeclId(data.teclId);
      if (data.subjName) state.courseName = data.subjName;
      ensureEpisodeListLoaded();
      recomputeSelection(true);
      syncCoursePreferenceFromServer();
      scheduleUiSync();
      return;
    }

    if (urlStr.includes('/v1/course/ai/ppt') && Array.isArray(data.docList)) {
      const cid = Number(parseQueryParam(urlStr, 'courseId') || 0);
      if (cid) {
        state.pptCache.set(cid, data.docList);
      }
      return;
    }

    const teclMatch = urlStr.match(/\/v1\/(?:teachingClass|tecl-ai-show-config)\/(\d+)/);
    if (teclMatch && !state.teclId) {
      setCurrentTeclId(teclMatch[1]);
      ensureEpisodeListLoaded();
    }
  }

  // Passive interception of both fetch and XHR at document-start.
  const origFetch = win.fetch ? win.fetch.bind(win) : null;
  if (origFetch) {
    win.fetch = function (input, init) {
      const urlStr = typeof input === 'string' ? input : (input && input.url) || '';
      if (urlStr.includes(API_PREFIX)) {
        if (init && init.headers) extractHeadersFromInit(init.headers);
        if (input && typeof input === 'object' && input.headers) extractHeadersFromInit(input.headers);
      }
      const p = origFetch(input, init);
      if (urlStr.includes(API_PREFIX)) {
        p.then((resp) => {
          if (!resp || !resp.ok) return;
          resp.clone().json().then((j) => handleInterceptedJson(urlStr, j)).catch(() => {});
        }).catch(() => {});
      }
      return p;
    };
  }

  const OrigXHR = win.XMLHttpRequest;
  if (OrigXHR) {
    const origOpen = OrigXHR.prototype.open;
    const origSetHeader = OrigXHR.prototype.setRequestHeader;
    const origSend = OrigXHR.prototype.send;

    OrigXHR.prototype.open = function (method, url) {
      this._sjtuUrl = String(url || '');
      return origOpen.apply(this, arguments);
    };

    OrigXHR.prototype.setRequestHeader = function (name, value) {
      if (this._sjtuUrl && this._sjtuUrl.includes(API_PREFIX)) {
        captureHeader(name, value);
      }
      return origSetHeader.apply(this, arguments);
    };

    OrigXHR.prototype.send = function () {
      if (this._sjtuUrl && this._sjtuUrl.includes(API_PREFIX)) {
        this.addEventListener('load', () => {
          try {
            if (this.status >= 200 && this.status < 300 && this.responseText) {
              const json = JSON.parse(this.responseText);
              handleInterceptedJson(this._sjtuUrl, json);
            }
          } catch {
            // Ignore non-JSON payloads.
          }
        });
      }
      return origSend.apply(this, arguments);
    };
  }

  let episodeFetchInFlight = false;
  async function ensureEpisodeListLoaded() {
    if (state.episodes.length > 0 || !state.teclId || episodeFetchInFlight) return;
    episodeFetchInFlight = true;
    try {
      const url = `/jy-application-resourcemanage/v1/subject_vod_list_new?page.pageIndex=1&page.pageSize=1000&teclIds=${state.teclId}&page.orders%5B0%5D.asc=false&page.orders%5B0%5D.field=courBeginTime&schoolOpenStatusFlag=false`;
      const json = await sjtuFetchJson(url);
      if (json && json.data && Array.isArray(json.data.records)) {
        state.episodes = json.data.records;
        recomputeSelection(true);
        scheduleUiSync();
      }
    } catch {
      // Passive capture may still populate this.
    } finally {
      episodeFetchInFlight = false;
    }
  }

  async function sjtuFetchJson(pathOrUrl) {
    const resp = await (origFetch || win.fetch)(pathOrUrl, {
      method: 'GET',
      credentials: 'include',
      headers: Object.assign({ Accept: 'application/json' }, state.headers),
    });
    if (!resp.ok) {
      throw new Error(`HTTP ${resp.status} for ${pathOrUrl}`);
    }
    return resp.json();
  }

  function groupLectureSessions(records) {
    const sorted = records
      .slice()
      .filter((r) => r && r.id)
      .sort((a, b) => {
        const ta = parseDateTimeMs(a.courBeginTime);
        const tb = parseDateTimeMs(b.courBeginTime);
        if (Number.isFinite(ta) && Number.isFinite(tb) && ta !== tb) return ta - tb;
        const na = Number(a.letiNumber || 0);
        const nb = Number(b.letiNumber || 0);
        if (na !== nb) return na - nb;
        return Number(a.id) - Number(b.id);
      });

    const groups = [];
    for (const ep of sorted) {
      const curDate = formatDatePart(ep.courBeginTime);
      const curStart = parseDateTimeMs(ep.courBeginTime);
      const prevGroup = groups[groups.length - 1];
      if (!prevGroup) {
        groups.push([ep]);
        continue;
      }
      const prevEp = prevGroup[prevGroup.length - 1];
      const prevDate = formatDatePart(prevEp.courBeginTime);
      const prevEnd = parseDateTimeMs(prevEp.courEndTime || prevEp.courBeginTime);
      const sameDay = Boolean(curDate && prevDate && curDate === prevDate);
      const gapOk = Number.isFinite(curStart) && Number.isFinite(prevEnd)
        ? (curStart - prevEnd) <= GROUP_MAX_GAP_MS
        : false;
      if (sameDay && gapOk) {
        prevGroup.push(ep);
      } else {
        groups.push([ep]);
      }
    }
    return groups;
  }

  function getActiveLectureGroup() {
    if (state.episodes.length === 0) {
      const fallback = state.vodCache.get(state.currentCourseId);
      if (fallback) {
        return [{
          id: Number(fallback.id),
          subjName: fallback.subjName || state.courseName,
          courBeginTime: fallback.courBeginTime ? new Date(fallback.courBeginTime).toISOString() : '',
          courEndTime: fallback.courEndTime ? new Date(fallback.courEndTime).toISOString() : '',
          letiNumber: fallback.letiNumber,
          letiName: fallback.letiName || '',
        }];
      }
      return [];
    }
    const groups = groupLectureSessions(state.episodes);
    if (state.currentCourseId) {
      const hit = groups.find((g) => g.some((ep) => Number(ep.id) === Number(state.currentCourseId)));
      if (hit) return hit;
    }
    return groups[groups.length - 1] || [];
  }

  function recomputeSelection(resetToCurrentGroup) {
    if (!resetToCurrentGroup && state.selectedIds.size > 0) return;
    const group = getActiveLectureGroup();
    if (group.length > 0) {
      state.selectedIds = new Set(group.map((ep) => Number(ep.id)));
    } else if (state.currentCourseId) {
      state.selectedIds = new Set([Number(state.currentCourseId)]);
    }
  }

  function localRequest(method, path, body, timeoutMs = 4000) {
    return new Promise((resolve, reject) => {
      GM_xmlhttpRequest({
        method,
        url: `${LOCAL_API}${path}`,
        headers: body !== undefined ? { 'Content-Type': 'application/json' } : {},
        data: body !== undefined ? JSON.stringify(body) : undefined,
        timeout: timeoutMs,
        onload(resp) {
          try {
            const parsed = resp.responseText ? JSON.parse(resp.responseText) : {};
            if (resp.status >= 200 && resp.status < 300) {
              resolve(parsed);
            } else {
              reject(new Error(parsed.detail || `本地服务返回 HTTP ${resp.status}`));
            }
          } catch (err) {
            reject(err);
          }
        },
        onerror() {
          reject(new Error('无法连接到本机伴侣服务 (127.0.0.1:18732)'));
        },
        ontimeout() {
          reject(new Error('连接本机伴侣服务超时'));
        },
      });
    });
  }

  function triggerProtocolWake() {
    const a = document.createElement('a');
    a.href = WAKE_URI;
    a.style.display = 'none';
    document.body.appendChild(a);
    a.click();
    a.remove();
  }

  async function ensureLocalServerAwake() {
    try {
      const h = await localRequest('GET', '/health', undefined, 450);
      state.apiKeyConfigured = Boolean(h.api_key_configured);
      return h;
    } catch {
      setStatus('正在通过 sjtu-mpv:// 唤醒本机伴侣服务...', 'info', 8);
      triggerProtocolWake();
      const deadline = Date.now() + 10000;
      while (Date.now() < deadline) {
        await new Promise((r) => setTimeout(r, 300));
        try {
          const h = await localRequest('GET', '/health', undefined, 500);
          state.apiKeyConfigured = Boolean(h.api_key_configured);
          return h;
        } catch {
          // Keep waiting for pythonw.exe startup.
        }
      }
      throw new Error('唤醒超时：请确认已运行 install_protocol.ps1 注册 sjtu-mpv:// 协议。');
    }
  }

  async function syncCoursePreferenceFromServer() {
    if (!state.teclId) return;
    try {
      await localRequest('GET', '/health', undefined, 350);
      const pref = await localRequest('GET', `/api/courses/${state.teclId}`, undefined, 1000);
      if (pref.preferred_view === 'ppt' || pref.preferred_view === 'camera') {
        saveLocalCourseView(state.teclId, pref.preferred_view);
        if (!state.viewManuallySelected) {
          state.preferredView = pref.preferred_view;
        }
      }
      if (Array.isArray(pref.glossary)) {
        state.glossaryText = pref.glossary.join('\n');
      }
      state.extraPrompt = pref.extra_prompt || '';
      scheduleUiSync();
    } catch {
      // Service is asleep right now; local cache keeps UI aligned until wake.
    }
  }

  async function fetchSegmentPayload(ep) {
    const cid = Number(ep.id);
    const vodResp = await sjtuFetchJson(`/jy-application-resourcemanage/v1/course_vod_urls_new?courseId=${cid}`);
    const vodData = vodResp && vodResp.data ? vodResp.data : null;
    if (!vodData || !Array.isArray(vodData.courseVodViewList) || vodData.courseVodViewList.length === 0) {
      throw new Error(`第 ${ep.letiNumber || cid} 节未获取到有效视频流地址`);
    }
    state.vodCache.set(cid, vodData);

    let pptList = state.pptCache.get(cid);
    if (!pptList) {
      try {
        const pptResp = await sjtuFetchJson(`/jy-application-resourcemanage/v1/course/ai/ppt?courseId=${cid}`);
        pptList = (pptResp && pptResp.data && Array.isArray(pptResp.data.docList)) ? pptResp.data.docList : [];
        state.pptCache.set(cid, pptList);
      } catch {
        pptList = [];
      }
    }

    const numLabel = ep.letiNumber != null ? `第${ep.letiNumber}节` : (vodData.letiName || `Section ${cid}`);
    const beginStr = typeof ep.courBeginTime === 'string' ? ep.courBeginTime : '';
    const endStr = typeof ep.courEndTime === 'string' ? ep.courEndTime : '';

    return {
      course_id: cid,
      title: `${formatDatePart(beginStr)} ${numLabel}`.trim(),
      begin_time: beginStr,
      end_time: endStr,
      leti_number: ep.letiNumber != null ? Number(ep.letiNumber) : (vodData.letiNumber ?? null),
      streams: vodData.courseVodViewList.map((s) => ({
        view_num: Number(s.viewNum),
        url: s.url,
        vod_time: s.vodTime != null ? Number(s.vodTime) : null,
      })),
      ppt_slices: (pptList || []).map((sl) => ({
        seek_sec: Number(sl.imageSeekTime || 0),
        ocr_text: String(sl.ocrText || ''),
        keywords: Array.isArray(sl.textKeywordList) ? sl.textKeywordList.map(String) : [],
      })),
    };
  }

  function getSelectedOrderedEpisodes() {
    const byId = new Map();
    for (const ep of state.episodes) {
      byId.set(Number(ep.id), ep);
    }
    if (state.currentCourseId && !byId.has(Number(state.currentCourseId))) {
      const vod = state.vodCache.get(Number(state.currentCourseId));
      byId.set(Number(state.currentCourseId), {
        id: Number(state.currentCourseId),
        courBeginTime: vod && vod.courBeginTime ? new Date(vod.courBeginTime).toISOString() : '',
        courEndTime: vod && vod.courEndTime ? new Date(vod.courEndTime).toISOString() : '',
        letiNumber: vod ? vod.letiNumber : null,
        subjName: state.courseName,
      });
    }
    const chosen = [];
    for (const id of state.selectedIds) {
      const ep = byId.get(Number(id));
      if (ep) chosen.push(ep);
    }
    return groupLectureSessions(chosen).flat();
  }

  async function onPlayClicked() {
    if (state.busy) return;
    const selected = getSelectedOrderedEpisodes();
    if (selected.length === 0) {
      setStatus('未检测到当前课程节次，请等待页面加载完成或刷新重试。', 'error', 0);
      return;
    }

    pauseAllWebMedia();
    state.busy = true;
    if (state.pollTimer) {
      clearInterval(state.pollTimer);
      state.pollTimer = null;
    }
    scheduleUiSync();

    try {
      const health = await ensureLocalServerAwake();
      if (!health.api_key_configured) {
        setStatus('警告：尚未在 config.yaml 配置 dashscope.api_key（仅缓存可播，新转录将报错）。', 'warn', 12);
      }

      if (state.teclId) {
        try {
          const pref = await localRequest('GET', `/api/courses/${state.teclId}`, undefined, 1000);
          if (pref.preferred_view === 'ppt' || pref.preferred_view === 'camera') {
            saveLocalCourseView(state.teclId, pref.preferred_view);
            if (!state.viewManuallySelected) {
              state.preferredView = pref.preferred_view;
              scheduleUiSync();
            }
          }
          if (Array.isArray(pref.glossary)) state.glossaryText = pref.glossary.join('\n');
          state.extraPrompt = pref.extra_prompt || '';
        } catch {
          // Non-fatal.
        }
      }

      setStatus(`正在拉取 ${selected.length} 节视频直链与 PPT 切片...`, 'info', 25);
      const segments = [];
      for (const ep of selected) {
        segments.push(await fetchSegmentPayload(ep));
      }

      const teclIdToUse = Number(state.teclId || segments[0].course_id);
      if (state.viewManuallySelected) {
        saveLocalCourseView(teclIdToUse, state.preferredView);
      }

      const payload = {
        tecl_id: teclIdToUse,
        course_name: state.courseName || 'SJTU 课堂实录',
        preferred_view: state.viewManuallySelected ? state.preferredView : null,
        force_retranscribe: state.forceRetranscribe,
        segments,
      };

      const res = await localRequest('POST', '/api/play', payload, 5000);
      setStatus('正在唤醒本地 mpv...', 'info', 70);
      startJobPolling(res.job_id);
    } catch (err) {
      state.busy = false;
      setStatus(err && err.message ? err.message : String(err), 'error', 0);
    }
  }

  function startJobPolling(jobId) {
    if (state.pollTimer) clearInterval(state.pollTimer);
    let closing = false;
    const pollOnce = async () => {
      if (closing) return;
      try {
        const job = await localRequest('GET', `/api/jobs/${jobId}`, undefined, 2500);

        if (job.state === 'failed') {
          clearInterval(state.pollTimer);
          state.pollTimer = null;
          state.busy = false;
          setStatus(job.error || job.message || '处理失败', 'error', 0);
          return;
        }
        if (job.state === 'canceled') {
          clearInterval(state.pollTimer);
          state.pollTimer = null;
          state.busy = false;
          setStatus(job.message || '已停止后台转录', 'warn', 0);
          return;
        }
        if (job.mpv_launched || job.state === 'playing' || job.state === 'completed') {
          closing = true;
          clearInterval(state.pollTimer);
          state.pollTimer = null;
          pauseAllWebMedia();
          setStatus('mpv 已启动，正在关闭本页...', 'success', 100);
          setTimeout(() => {
            try {
              window.close();
            } catch {
              // Ignore if blocked.
            }
            setTimeout(() => {
              state.busy = false;
              setStatus('mpv 已接管播放（请在 mpv 内查看进度），可关闭本页', 'success', 100);
            }, 800);
          }, 250);
          return;
        }

        const pct = job.state === 'probing' ? 50 : 75;
        setStatus(job.message || '正在准备唤醒 mpv...', 'info', pct);
      } catch {
        // Ignore transient poll glitches.
      }
    };
    pollOnce();
    state.pollTimer = setInterval(pollOnce, 400);
  }

  function setStatus(text, type, pct) {
    state.statusText = text;
    state.statusType = type;
    state.progressPct = pct;
    scheduleUiSync();
  }

  async function openGlossaryModal() {
    if (!state.teclId) {
      setStatus('请先等待课程 ID 加载完成。', 'warn', 0);
      return;
    }
    try {
      await ensureLocalServerAwake();
      const pref = await localRequest('GET', `/api/courses/${state.teclId}`, undefined, 2000);
      state.glossaryText = Array.isArray(pref.glossary) ? pref.glossary.join('\n') : '';
      state.extraPrompt = pref.extra_prompt || '';
      if (pref.preferred_view === 'ppt' || pref.preferred_view === 'camera') {
        saveLocalCourseView(state.teclId, pref.preferred_view);
        if (!state.viewManuallySelected) {
          state.preferredView = pref.preferred_view;
        }
      }
    } catch (err) {
      setStatus(err.message || String(err), 'error', 0);
      return;
    }

    const old = document.getElementById('sjtu-mpv-glossary-modal');
    if (old) old.remove();

    const modal = document.createElement('div');
    modal.id = 'sjtu-mpv-glossary-modal';
    modal.innerHTML = `
      <div class="sjtu-mpv-modal-card">
        <div class="sjtu-mpv-modal-header">
          <span>📝 课程术语表与提示词 · ${escapeHtml(state.courseName || String(state.teclId))}</span>
          <button type="button" class="sjtu-mpv-icon-btn" id="sjtu-mpv-modal-close">✕</button>
        </div>
        <label class="sjtu-mpv-label">课程专属术语（每行一个，用于纠正 ASR 专有名词与统一译法）</label>
        <textarea id="sjtu-mpv-glossary-input" rows="7" placeholder="例如：\nEigenvalue = 特征值\nKirchhoff's Law\nCMOS"></textarea>
        <label class="sjtu-mpv-label">额外翻译提示（可选）</label>
        <textarea id="sjtu-mpv-prompt-input" rows="3" placeholder="例如：保留数学符号英文原名，中文尽量精炼"></textarea>
        <div class="sjtu-mpv-modal-actions">
          <button type="button" class="sjtu-mpv-btn-secondary" id="sjtu-mpv-modal-cancel">取消</button>
          <button type="button" class="sjtu-mpv-btn-primary" id="sjtu-mpv-modal-save">保存到本机 courses.yaml</button>
        </div>
      </div>
    `;
    document.body.appendChild(modal);
    modal.querySelector('#sjtu-mpv-glossary-input').value = state.glossaryText;
    modal.querySelector('#sjtu-mpv-prompt-input').value = state.extraPrompt;

    const close = () => modal.remove();
    modal.querySelector('#sjtu-mpv-modal-close').addEventListener('click', close);
    modal.querySelector('#sjtu-mpv-modal-cancel').addEventListener('click', close);
    modal.querySelector('#sjtu-mpv-modal-save').addEventListener('click', async () => {
      const glossaryLines = modal
        .querySelector('#sjtu-mpv-glossary-input')
        .value.split(/\r?\n/)
        .map((s) => s.trim())
        .filter(Boolean);
      const extraPrompt = modal.querySelector('#sjtu-mpv-prompt-input').value.trim();
      try {
        const updateBody = {
          name: state.courseName,
          glossary: glossaryLines,
          extra_prompt: extraPrompt,
        };
        if (state.viewManuallySelected) {
          updateBody.preferred_view = state.preferredView;
          saveLocalCourseView(state.teclId, state.preferredView);
        }
        await localRequest('PUT', `/api/courses/${state.teclId}`, updateBody);
        state.glossaryText = glossaryLines.join('\n');
        state.extraPrompt = extraPrompt;
        setStatus(`已保存 ${glossaryLines.length} 条课程术语`, 'success', 0);
        close();
      } catch (err) {
        setStatus(err.message || String(err), 'error', 0);
      }
    });
  }

  function escapeHtml(str) {
    return String(str || '')
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;');
  }

  let stylesInjected = false;
  function injectStyles() {
    if (stylesInjected) return;
    stylesInjected = true;
    const css = `
      #sjtu-mpv-panel {
        position: fixed;
        right: 20px;
        bottom: 20px;
        width: 340px;
        z-index: 99999;
        background: rgba(15, 28, 48, 0.96);
        color: #e6eef8;
        border: 1px solid rgba(90, 155, 255, 0.35);
        border-radius: 12px;
        box-shadow: 0 12px 32px rgba(0, 0, 0, 0.45);
        font-family: "Microsoft YaHei UI", system-ui, sans-serif;
        font-size: 13px;
        backdrop-filter: blur(8px);
        overflow: hidden;
      }
      #sjtu-mpv-panel.minimized { width: auto; }
      .sjtu-mpv-head {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 10px 12px;
        background: linear-gradient(90deg, #163156, #1b4072);
        font-weight: 600;
        gap: 10px;
      }
      .sjtu-mpv-body { padding: 12px; display: flex; flex-direction: column; gap: 10px; }
      #sjtu-mpv-panel.minimized .sjtu-mpv-body { display: none; }
      .sjtu-mpv-summary {
        background: rgba(255, 255, 255, 0.06);
        border-radius: 8px;
        padding: 8px 10px;
        line-height: 1.45;
      }
      .sjtu-mpv-summary-title { font-weight: 600; color: #fff; }
      .sjtu-mpv-summary-sub { font-size: 12px; color: #9fc0e8; margin-top: 2px; }
      .sjtu-mpv-toggle-link {
        background: none; border: none; color: #67b2ff; padding: 0;
        cursor: pointer; font-size: 12px; text-align: left;
      }
      .sjtu-mpv-ep-list {
        max-height: 160px; overflow-y: auto; display: flex; flex-direction: column; gap: 4px;
        background: rgba(0, 0, 0, 0.25); border-radius: 6px; padding: 6px 8px;
      }
      .sjtu-mpv-ep-item { display: flex; align-items: center; gap: 6px; font-size: 12px; cursor: pointer; }
      .sjtu-mpv-view-group { display: grid; grid-template-columns: 1fr 1fr; gap: 6px; }
      .sjtu-mpv-seg-btn {
        border: 1px solid rgba(120, 170, 240, 0.3);
        background: rgba(255, 255, 255, 0.05);
        color: #c9ddf5;
        border-radius: 6px;
        padding: 6px 8px;
        cursor: pointer;
        font-size: 12px;
      }
      .sjtu-mpv-seg-btn.active {
        background: #2563eb;
        border-color: #60a5fa;
        color: #fff;
        font-weight: 600;
      }
      .sjtu-mpv-actions { display: flex; gap: 8px; }
      .sjtu-mpv-btn-primary {
        flex: 1;
        background: linear-gradient(135deg, #2563eb, #1d4ed8);
        color: #fff;
        border: none;
        border-radius: 8px;
        padding: 9px 12px;
        font-weight: 600;
        cursor: pointer;
      }
      .sjtu-mpv-btn-primary:disabled { opacity: 0.6; cursor: not-allowed; }
      .sjtu-mpv-btn-secondary {
        background: rgba(255, 255, 255, 0.1);
        color: #e6eef8;
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 8px;
        padding: 8px 10px;
        cursor: pointer;
      }
      .sjtu-mpv-icon-btn {
        background: rgba(255, 255, 255, 0.12);
        color: #fff;
        border: none;
        border-radius: 6px;
        width: 24px;
        height: 24px;
        cursor: pointer;
      }
      .sjtu-mpv-meta-row {
        display: flex; align-items: center; justify-content: space-between;
        font-size: 12px; color: #b7cde8;
      }
      .sjtu-mpv-status {
        font-size: 12px; line-height: 1.4; padding: 6px 8px; border-radius: 6px;
        background: rgba(255, 255, 255, 0.06);
      }
      .sjtu-mpv-status.error { background: rgba(220, 38, 38, 0.22); color: #fecaca; }
      .sjtu-mpv-status.warn { background: rgba(217, 119, 6, 0.24); color: #fde68a; }
      .sjtu-mpv-status.success { background: rgba(22, 163, 74, 0.22); color: #bbf7d0; }
      .sjtu-mpv-bar-wrap { height: 4px; background: rgba(255, 255, 255, 0.1); border-radius: 4px; overflow: hidden; }
      .sjtu-mpv-bar { height: 100%; background: #38bdf8; transition: width 0.25s ease; }
      #sjtu-mpv-glossary-modal {
        position: fixed; inset: 0; z-index: 100000;
        background: rgba(5, 10, 20, 0.7);
        display: flex; align-items: center; justify-content: center;
      }
      .sjtu-mpv-modal-card {
        width: 440px; max-width: 92vw;
        background: #102038; color: #e6eef8;
        border: 1px solid rgba(90, 155, 255, 0.4);
        border-radius: 12px; padding: 16px;
        display: flex; flex-direction: column; gap: 10px;
      }
      .sjtu-mpv-modal-header { display: flex; justify-content: space-between; align-items: center; font-weight: 600; }
      .sjtu-mpv-label { font-size: 12px; color: #9fc0e8; }
      .sjtu-mpv-modal-card textarea {
        width: 100%; box-sizing: border-box; border-radius: 6px; padding: 8px;
        background: #081222; color: #fff; border: 1px solid rgba(120, 170, 240, 0.3);
        font-family: inherit; font-size: 13px;
      }
      .sjtu-mpv-modal-actions { display: flex; justify-content: flex-end; gap: 8px; margin-top: 4px; }
    `;
    if (typeof GM_addStyle === 'function') {
      GM_addStyle(css);
    } else {
      const style = document.createElement('style');
      style.textContent = css;
      document.head.appendChild(style);
    }
  }

  let renderQueued = false;
  function scheduleUiSync() {
    if (renderQueued) return;
    renderQueued = true;
    setTimeout(() => {
      renderQueued = false;
      renderPanel();
    }, 60);
  }

  function renderPanel() {
    if (!document.body) return;
    const existing = document.getElementById('sjtu-mpv-panel');
    if (!isPlayCenter()) {
      if (existing) existing.remove();
      return;
    }
    injectStyles();

    const panel = existing || document.createElement('div');
    panel.id = 'sjtu-mpv-panel';
    panel.className = state.minimized ? 'minimized' : '';

    const selected = getSelectedOrderedEpisodes();
    const dateLabel = selected.length > 0 ? formatDatePart(selected[0].courBeginTime) : '';
    const nums = selected.map((e) => e.letiNumber).filter((n) => n != null);
    const rangeLabel = nums.length > 0
      ? (nums.length > 1 ? `第${nums[0]}-${nums[nums.length - 1]}节` : `第${nums[0]}节`)
      : '';
    const groupTitle = `${state.courseName || '当前课程'}`;
    const groupSub = selected.length > 0
      ? `${dateLabel} ${rangeLabel} (已选 ${selected.length} 节)`.trim()
      : '等待页面加载节次信息...';

    const allList = state.episodes.length > 0
      ? state.episodes.slice().sort((a, b) => parseDateTimeMs(a.courBeginTime) - parseDateTimeMs(b.courBeginTime))
      : selected;

    panel.innerHTML = `
      <div class="sjtu-mpv-head">
        <span>🎬 mpv 双语流式播放</span>
        <button type="button" class="sjtu-mpv-icon-btn" id="sjtu-mpv-min">${state.minimized ? '＋' : '－'}</button>
      </div>
      <div class="sjtu-mpv-body">
        <div class="sjtu-mpv-summary">
          <div class="sjtu-mpv-summary-title">${escapeHtml(groupTitle)}</div>
          <div class="sjtu-mpv-summary-sub">${escapeHtml(groupSub)}</div>
        </div>
        <button type="button" class="sjtu-mpv-toggle-link" id="sjtu-mpv-toggle-eps">
          ${state.showEpisodes ? '▾ 收起节次列表' : '▸ 手动增减连堂节次'}
        </button>
        ${state.showEpisodes ? `
          <div class="sjtu-mpv-ep-list">
            ${allList.map((ep) => `
              <label class="sjtu-mpv-ep-item">
                <input type="checkbox" class="sjtu-mpv-ep-cb" value="${ep.id}" ${state.selectedIds.has(Number(ep.id)) ? 'checked' : ''} />
                <span>${escapeHtml(formatPeriodLabel(ep))}</span>
              </label>
            `).join('')}
          </div>
        ` : ''}
        <div class="sjtu-mpv-view-group">
          <button type="button" class="sjtu-mpv-seg-btn ${state.preferredView === 'ppt' ? 'active' : ''}" data-view="ppt">PPT 屏录</button>
          <button type="button" class="sjtu-mpv-seg-btn ${state.preferredView === 'camera' ? 'active' : ''}" data-view="camera">教室画面</button>
        </div>
        <div class="sjtu-mpv-actions">
          <button type="button" class="sjtu-mpv-btn-primary" id="sjtu-mpv-play" ${state.busy ? 'disabled' : ''}>
            ${state.busy ? '正在启动 mpv...' : '▶ 用 mpv 双语播放'}
          </button>
          <button type="button" class="sjtu-mpv-btn-secondary" id="sjtu-mpv-glossary">📝 术语表</button>
        </div>
        <div class="sjtu-mpv-meta-row">
          <label style="display:flex;align-items:center;gap:5px;cursor:pointer;">
            <input type="checkbox" id="sjtu-mpv-force" ${state.forceRetranscribe ? 'checked' : ''} />
            <span>强制重新转录</span>
          </label>
        </div>
        ${!state.apiKeyConfigured ? `
          <div class="sjtu-mpv-status warn">⚠️ 未检测到 DashScope API Key，请在 D:\\Program Files\\sjtu-mpv-sub\\config.yaml 配置 dashscope.api_key</div>
        ` : ''}
        <div class="sjtu-mpv-status ${state.statusType}">${escapeHtml(state.statusText)}</div>
        ${state.progressPct > 0 ? `
          <div class="sjtu-mpv-bar-wrap"><div class="sjtu-mpv-bar" style="width:${state.progressPct}%"></div></div>
        ` : ''}
      </div>
    `;

    if (!existing) {
      document.body.appendChild(panel);
    }

    panel.querySelector('#sjtu-mpv-min').addEventListener('click', () => {
      state.minimized = !state.minimized;
      renderPanel();
    });
    panel.querySelector('#sjtu-mpv-toggle-eps').addEventListener('click', () => {
      state.showEpisodes = !state.showEpisodes;
      renderPanel();
    });
    panel.querySelectorAll('.sjtu-mpv-ep-cb').forEach((cb) => {
      cb.addEventListener('change', (ev) => {
        const id = Number(ev.target.value);
        if (ev.target.checked) state.selectedIds.add(id);
        else state.selectedIds.delete(id);
        renderPanel();
      });
    });
    panel.querySelectorAll('.sjtu-mpv-seg-btn').forEach((btn) => {
      btn.addEventListener('click', () => {
        state.preferredView = btn.dataset.view === 'camera' ? 'camera' : 'ppt';
        state.viewManuallySelected = true;
        saveLocalCourseView(state.teclId, state.preferredView);
        renderPanel();
      });
    });
    panel.querySelector('#sjtu-mpv-force').addEventListener('change', (ev) => {
      state.forceRetranscribe = Boolean(ev.target.checked);
    });
    panel.querySelector('#sjtu-mpv-play').addEventListener('click', onPlayClicked);
    panel.querySelector('#sjtu-mpv-glossary').addEventListener('click', openGlossaryModal);
  }

  function handleRouteChange() {
    state.allowWebVideoPlay = false;
    scheduleUiSync();
  }

  const origPushState = win.history.pushState.bind(win.history);
  win.history.pushState = function () {
    const res = origPushState.apply(this, arguments);
    handleRouteChange();
    return res;
  };
  const origReplaceState = win.history.replaceState.bind(win.history);
  win.history.replaceState = function () {
    const res = origReplaceState.apply(this, arguments);
    handleRouteChange();
    return res;
  };
  window.addEventListener('hashchange', handleRouteChange);
  window.addEventListener('popstate', handleRouteChange);
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
      pauseAllWebMedia();
      scheduleUiSync();
    });
  } else {
    pauseAllWebMedia();
    scheduleUiSync();
  }
})();
