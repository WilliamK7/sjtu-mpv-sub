-- sjtu_sub_hud.lua: In-player subtitle transcription progress HUD & controls for sjtu-mpv-sub.
local mp = require("mp")
local utils = require("mp.utils")

local overlay = mp.create_osd_overlay("ass-events")
overlay.res_x = 1920
overlay.res_y = 1080

local state = {
    phase = "idle", -- idle | detecting | transcribing | completed | canceled
    total_sec = 0,
    covered = {},
    in_flight = {},
    urgent = nil,
    covered_summary = "",
    pct = 0,
    headline = "",
    pos = nil,
}

-- nil = auto (show while active, hide on completed/canceled); true = force show; false = force hide
local user_override = nil
local last_pos_render_at = 0

local function esc_ass(str)
    if not str then return "" end
    local s = tostring(str)
    s = s:gsub("\\", "\\\\")
    s = s:gsub("{", "｛")
    s = s:gsub("}", "｝")
    return s
end

local function fmt_time(sec)
    local s = math.max(0, math.floor((tonumber(sec) or 0) + 0.5))
    local h = math.floor(s / 3600)
    local m = math.floor((s % 3600) / 60)
    local rem = s % 60
    if h > 0 then
        return string.format("%d:%02d:%02d", h, m, rem)
    end
    return string.format("%02d:%02d", m, rem)
end

local function pos_in_ranges(pos, ranges)
    if not pos or type(ranges) ~= "table" then return false end
    for _, seg in ipairs(ranges) do
        if type(seg) == "table" and #seg >= 2 then
            local a = tonumber(seg[1]) or 0
            local b = tonumber(seg[2]) or 0
            if pos >= (a - 0.5) and pos <= (b + 0.5) then
                return true
            end
        end
    end
    return false
end

local function should_show()
    if state.phase == "idle" then
        return false
    end
    if user_override ~= nil then
        return user_override
    end
    return state.phase == "detecting" or state.phase == "transcribing"
end

local function draw_rect(x1, y1, x2, y2)
    return string.format("m %d %d l %d %d %d %d %d %d", x1, y1, x2, y1, x2, y2, x1, y2)
end

local function clamp(val, lo, hi)
    if val < lo then return lo end
    if val > hi then return hi end
    return val
end

local function render()
    if not should_show() then
        overlay:remove()
        return
    end

    local bx1, by1, bx2, by2 = 1344, 54, 1896, 140
    local pad_x = 1364
    local bar_x1, bar_x2 = 1364, 1876
    local bar_w = bar_x2 - bar_x1
    local bar_y1, bar_y2 = 97, 105

    local pos = state.pos
    local covered_here = pos and pos_in_ranges(pos, state.covered)
    local is_uncovered_active = (state.phase == "transcribing") and (pos ~= nil) and (not covered_here)

    local bg_alpha = is_uncovered_active and "1C" or "48"
    local border_color = is_uncovered_active and "&H38B8F5&" or "&H8C5B3B&"

    local headline = state.headline or ""
    if state.phase == "detecting" then
        if headline == "" then
            headline = "🔍 正在智能定位开课时间..."
        end
    elseif state.phase == "transcribing" then
        if is_uncovered_active then
            if type(state.urgent) == "table" and #state.urgent >= 2 and pos >= (tonumber(state.urgent[1]) - 5) and pos <= (tonumber(state.urgent[2]) + 5) then
                headline = string.format("⏳ 正在优先转录当前位置 (%s–%s)...", fmt_time(state.urgent[1]), fmt_time(state.urgent[2]))
            else
                headline = string.format("⏳ 当前位置 (%s) 暂无字幕，优先排队中...", fmt_time(pos))
            end
        elseif headline == "" then
            local sum = (state.covered_summary and state.covered_summary ~= "") and state.covered_summary or "准备中"
            headline = string.format("字幕后台补齐 · 已就绪 %s (%d%%)", sum, tonumber(state.pct) or 0)
        end
    elseif state.phase == "completed" then
        local sum = (state.covered_summary and state.covered_summary ~= "") and state.covered_summary or "全片"
        headline = string.format("✓ 字幕已就绪 · %s", sum)
    elseif state.phase == "canceled" then
        local sum = (state.covered_summary and state.covered_summary ~= "") and state.covered_summary or "无"
        headline = string.format("⏹ 已停止后台转录 · 已保留 %s", sum)
    end

    local text_color = is_uncovered_active and "&H73D8FF&" or "&HFFFFFF&"
    local hint_text = "Alt+s 显隐 · Alt+x 停止后台转录"
    if state.phase == "completed" or state.phase == "canceled" then
        hint_text = "Alt+s 隐藏面板"
    end

    local lines = {}
    -- Pill background + 1px accent border
    lines[#lines + 1] = string.format(
        "{\\an7\\pos(0,0)\\bord0\\shad0\\1c%s\\1a&H00&\\p1}%s{\\p0}",
        border_color,
        draw_rect(bx1 - 1, by1 - 1, bx2 + 1, by2 + 1)
    )
    lines[#lines + 1] = string.format(
        "{\\an7\\pos(0,0)\\bord0\\shad0\\1c&H28190F&\\1a&H%s&\\p1}%s{\\p0}",
        bg_alpha,
        draw_rect(bx1, by1, bx2, by2)
    )

    -- Headline
    lines[#lines + 1] = string.format(
        "{\\an7\\pos(%d,%d)\\fnMicrosoft YaHei UI\\fs21\\b1\\bord1\\shad0\\1c%s\\3c&H000000&\\1a&H00&}%s",
        pad_x, 62, text_color, esc_ass(headline)
    )

    -- Coverage timeline bar track
    lines[#lines + 1] = string.format(
        "{\\an7\\pos(0,0)\\bord0\\shad0\\1c&H50392A&\\1a&H20&\\p1}%s{\\p0}",
        draw_rect(bar_x1, bar_y1, bar_x2, bar_y2)
    )

    local total = tonumber(state.total_sec) or 0
    if total > 0 then
        -- Covered intervals (green)
        for _, seg in ipairs(state.covered or {}) do
            if type(seg) == "table" and #seg >= 2 then
                local sx = clamp(bar_x1 + math.floor((tonumber(seg[1]) / total) * bar_w), bar_x1, bar_x2)
                local ex = clamp(bar_x1 + math.ceil((tonumber(seg[2]) / total) * bar_w), bar_x1, bar_x2)
                if ex <= sx then ex = math.min(bar_x2, sx + 2) end
                lines[#lines + 1] = string.format(
                    "{\\an7\\pos(0,0)\\bord0\\shad0\\1c&H7CD152&\\1a&H10&\\p1}%s{\\p0}",
                    draw_rect(sx, bar_y1, ex, bar_y2)
                )
            end
        end
        -- In-flight intervals (amber)
        for _, seg in ipairs(state.in_flight or {}) do
            if type(seg) == "table" and #seg >= 2 then
                local sx = clamp(bar_x1 + math.floor((tonumber(seg[1]) / total) * bar_w), bar_x1, bar_x2)
                local ex = clamp(bar_x1 + math.ceil((tonumber(seg[2]) / total) * bar_w), bar_x1, bar_x2)
                if ex <= sx then ex = math.min(bar_x2, sx + 2) end
                lines[#lines + 1] = string.format(
                    "{\\an7\\pos(0,0)\\bord0\\shad0\\1c&H38B8F5&\\1a&H10&\\p1}%s{\\p0}",
                    draw_rect(sx, bar_y1, ex, bar_y2)
                )
            end
        end
        -- Playhead marker
        if pos ~= nil then
            local px = clamp(bar_x1 + math.floor((pos / total) * bar_w), bar_x1, bar_x2)
            local mx1 = clamp(px - 1, bar_x1, bar_x2 - 2)
            local mx2 = clamp(px + 2, bar_x1 + 2, bar_x2)
            lines[#lines + 1] = string.format(
                "{\\an7\\pos(0,0)\\bord0\\shad0\\1c&HFFFFFF&\\1a&H00&\\p1}%s{\\p0}",
                draw_rect(mx1, bar_y1 - 3, mx2, bar_y2 + 3)
            )
        end
    end

    -- Secondary hint
    lines[#lines + 1] = string.format(
        "{\\an7\\pos(%d,%d)\\fnMicrosoft YaHei UI\\fs16\\b0\\bord1\\shad0\\1c&HDFC1A9&\\3c&H000000&\\1a&H10&}%s",
        pad_x, 112, esc_ass(hint_text)
    )

    overlay.data = table.concat(lines, "\n")
    overlay:update()
end

mp.register_script_message("update", function(json_str)
    if not json_str or json_str == "" then return end
    local parsed = utils.parse_json(json_str)
    if type(parsed) ~= "table" then return end

    local prev_phase = state.phase
    state.phase = parsed.phase or state.phase
    state.total_sec = tonumber(parsed.total_sec) or state.total_sec
    state.covered = type(parsed.covered) == "table" and parsed.covered or state.covered
    state.in_flight = type(parsed.in_flight) == "table" and parsed.in_flight or {}
    state.urgent = type(parsed.urgent) == "table" and parsed.urgent or nil
    state.covered_summary = parsed.covered_summary or ""
    state.pct = tonumber(parsed.pct) or 0
    state.headline = parsed.headline or ""

    if (state.phase == "completed" or state.phase == "canceled") and prev_phase ~= state.phase then
        user_override = nil
    end

    render()
end)

mp.register_script_message("clear", function()
    state.phase = "idle"
    user_override = nil
    overlay:remove()
end)

mp.observe_property("time-pos", "number", function(_, val)
    if val == nil then return end
    local prev_cov = state.pos and pos_in_ranges(state.pos, state.covered)
    state.pos = val
    local now_cov = pos_in_ranges(val, state.covered)
    local now = mp.get_time()
    if (prev_cov ~= now_cov) or (now - last_pos_render_at >= 0.45) then
        last_pos_render_at = now
        if should_show() then
            render()
        end
    end
end)

mp.add_key_binding("Alt+s", "sjtu-sub-toggle-hud", function()
    if state.phase == "idle" then
        mp.osd_message("暂无字幕转录任务状态", 1800)
        return
    end
    local cur = should_show()
    user_override = not cur
    render()
    mp.osd_message(user_override and "已显示字幕进度面板" or "已隐藏字幕进度面板", 1500)
end)

mp.add_key_binding("Alt+x", "sjtu-sub-stop-bg", function()
    if state.phase == "completed" or state.phase == "canceled" or state.phase == "idle" then
        mp.osd_message("当前没有运行中的后台转录任务", 1800)
        return
    end
    mp.set_property_native("user-data/sjtu-sub/cancel", true)
    mp.osd_message("⏹ 正在停止后台字幕转录（已保留当前进度）...", 2500)
end)
