-- sjtu_thumb.lua: Lightweight Referer-aware remote thumbnailer for uosc + EDL lectures.
local mp = require("mp")
mp.options = require("mp.options")
mp.utils = require("mp.utils")

local options = {
	load = true,
	mpv_path = "mpv",
	referer = "",
	socket = "",
	tnpath = "",
	max_width = 320,
	max_height = 320,
	rescale = 0,
	sw_threads = 2,
	frequency = 0.15,
	min_seek_delta = 0.5,
	quit_after_inactivity = 120,
	overlay_id = 10,
	min_duration = 10,
	direct_io = true,
}
mp.options.read_options(options, "sjtu_thumb")

if options.load == false then
	mp.msg.info("sjtu_thumb disabled by option")
	return
end

local os_name = mp.get_property("platform")
local pid = mp.utils.getpid()

if options.tnpath == "" then
	if os_name == "windows" then
		local tmp = os.getenv("TEMP") or os.getenv("TMP") or "."
		options.tnpath = tmp .. "\\sjtu_thumb_" .. tostring(pid) .. ".out"
	else
		options.tnpath = "/tmp/sjtu_thumb_" .. tostring(pid) .. ".out"
	end
else
	options.tnpath = options.tnpath .. tostring(pid)
end

if options.socket == "" then
	if os_name == "windows" then
		options.socket = "sjtu_thumb_" .. tostring(pid)
	else
		options.socket = "/tmp/sjtu_thumb_" .. tostring(pid)
	end
else
	options.socket = options.socket .. tostring(pid)
end

-- Windows Named Pipe FFI setup (matches thumbfast / thumb_engine non-blocking pipe writes).
local winapi = { enabled = false }
if options.direct_io and os_name == "windows" then
	local ffi_ok, ffi = pcall(require, "ffi")
	local bit_ok, bit = pcall(require, "bit")
	if ffi_ok and bit_ok then
		pcall(ffi.cdef, [[
			void* __stdcall CreateFileW(const wchar_t *lpFileName, unsigned long dwDesiredAccess, unsigned long dwShareMode, void *lpSecurityAttributes, unsigned long dwCreationDisposition, unsigned long dwFlagsAndAttributes, void *hTemplateFile);
			bool __stdcall WriteFile(void *hFile, const void *lpBuffer, unsigned long nNumberOfBytesToWrite, unsigned long *lpNumberOfBytesWritten, void *lpOverlapped);
			bool __stdcall CloseHandle(void *hObject);
			bool __stdcall SetNamedPipeHandleState(void *hNamedPipe, unsigned long *lpMode, unsigned long *lpMaxCollectionCount, unsigned long *lpCollectDataTimeout);
			int __stdcall MultiByteToWideChar(unsigned int CodePage, unsigned long dwFlags, const char *lpMultiByteStr, int cbMultiByte, wchar_t *lpWideCharStr, int cchWideChar);
		]])
		local pipe_name = "\\\\.\\pipe\\" .. options.socket
		local utf16_len = ffi.C.MultiByteToWideChar(65001, 0, pipe_name, -1, nil, 0)
		if utf16_len > 0 then
			local wide = ffi.new("wchar_t[?]", utf16_len)
			if ffi.C.MultiByteToWideChar(65001, 0, pipe_name, -1, wide, utf16_len) > 0 then
				winapi.enabled = true
				winapi.ffi = ffi
				winapi.C = ffi.C
				winapi.socket_wc = wide
				winapi.GENERIC_WRITE = 0x40000000
				winapi.OPEN_EXISTING = 3
				winapi.INVALID_HANDLE_VALUE = ffi.cast("void*", -1)
				winapi.PIPE_NOWAIT = ffi.new("unsigned long[1]", 0x00000001)
				winapi._written = ffi.new("unsigned long[1]")
				winapi._flags = bit.bor(0x80000000, 0x20000000)
			end
		end
	end
end

local state = {
	file = nil,
	file_bytes = 0,
	spawned = false,
	disabled = false,
	dirty = false,
	x = nil,
	y = nil,
	last_x = nil,
	last_y = nil,
	last_seek_time = nil,
	pending_ipc_seek = false,
	effective_w = options.max_width,
	effective_h = options.max_height,
	real_w = nil,
	real_h = nil,
	last_real_w = nil,
	last_real_h = nil,
	show_thumbnail = false,
	has_vid = 0,
	last_has_vid = 0,
	auto_run = true,
	info_timer = nil,
	properties = {},
}

local preview_draw = nil
local preview_ass = mp.create_osd_overlay("ass-events")

local function move_file(from, to)
	if os_name == "windows" then
		os.remove(to)
	end
	return os.rename(from, to)
end

local function remove_thumbnail_files()
	if state.file then
		pcall(function() state.file:close() end)
		state.file = nil
		state.file_bytes = 0
	end
	os.remove(options.tnpath)
	os.remove(options.tnpath .. ".tmp")
	os.remove(options.tnpath .. ".bgra")
end

local function auto_ui_scale()
	local dw = mp.get_property_number("display-width", 0)
	local dh = mp.get_property_number("display-height", 0)
	if dw <= 0 or dh <= 0 then return 1 end
	local aspect = dw / dh
	if aspect <= 1 then return 1 end
	if aspect >= 2 then
		return tonumber(string.format("%.2f", dh / 1080))
	end
	if dw * dh > 2304000 then
		return tonumber(string.format("%.2f", math.sqrt(dw * dh / 2073600)))
	end
	return 1
end

local function calc_dimensions()
	local vp = state.properties["video-params"]
	local width = vp and vp["w"]
	local height = vp and vp["h"]
	if not width or not height or width <= 0 or height <= 0 then return end

	local rescale = tonumber(options.rescale) or 0
	local scale = rescale == 0 and (auto_ui_scale() or 1) or math.max(rescale, 1)
	local w, h
	if width / height > options.max_width / options.max_height then
		w = math.floor(options.max_width * scale + 0.5)
		h = math.floor(height / width * w + 0.5)
	else
		h = math.floor(options.max_height * scale + 0.5)
		w = math.floor(width / height * h + 0.5)
	end
	-- Keep even pixel dimensions for raw BGRA stride predictability.
	w = math.max(2, w - (w % 2))
	h = math.max(2, h - (h % 2))
	state.effective_w = w
	state.effective_h = h
end

local function real_res(req_w, req_h, filesize)
	local count = filesize / 4
	local dec = state.properties["video-dec-params"]
	if dec and ((dec["rotate"] or 0) % 180 == 90) then
		req_w, req_h = req_h, req_w
	end
	if req_w * req_h == count then
		return req_w, req_h
	end
	local threshold = 5
	local long_side, short_side = req_w, req_h
	if req_h > req_w then
		long_side, short_side = req_h, req_w
	end
	for a = short_side, short_side - threshold, -1 do
		if a > 0 and count % a == 0 then
			local b = count / a
			if math.abs(long_side - b) < threshold then
				if req_h < req_w then return b, a else return a, b end
			end
		end
	end
	return nil
end

local function check_new_thumb()
	local tmp = options.tnpath .. ".tmp"
	move_file(options.tnpath, tmp)
	local finfo = mp.utils.file_info(tmp)
	if not finfo or not finfo.size or finfo.size <= 0 then
		return nil
	end
	local w, h = real_res(state.effective_w, state.effective_h, finfo.size)
	if w and h then
		move_file(tmp, options.tnpath .. ".bgra")
		return w, h
	end
	return nil
end

local function broadcast_info(w, h)
	local dur = mp.get_property_number("duration", 0)
	local short_video = (options.min_duration > 0) and (dur > 0) and (dur <= options.min_duration)
	local vid_track = state.properties["current-tracks/video"]
	local image = vid_track and vid_track["image"]
	local albumart = image and vid_track["albumart"]
	local vid_prop = mp.get_property("vid")
	local has_vid = (state.has_vid ~= 0) or (vid_prop ~= nil and vid_prop ~= "no" and vid_prop ~= "false" and vid_prop ~= "0")

	state.disabled = (w or 0) == 0 or (h or 0) == 0 or
		(not has_vid) or
		albumart or
		(image and not albumart) or
		short_video or
		not state.auto_run

	if state.info_timer then
		state.info_timer:kill()
		state.info_timer = nil
	elseif (not has_vid) or not state.disabled then
		state.info_timer = mp.add_timeout(0.1, function()
			broadcast_info(state.real_w or state.effective_w, state.real_h or state.effective_h)
		end)
	end

	local json = mp.utils.format_json({
		width = w,
		height = h,
		disabled = state.disabled,
		available = true,
		socket = options.socket,
		tnpath = options.tnpath,
		overlay_id = options.overlay_id,
	})
	mp.command_native_async({"script-message", "sjtu_thumb-info", json}, function() end)
	mp.set_property_native("user-data/mpv/thumbnailer/enabled", not state.disabled)
end

local function run_ipc(command)
	if not state.spawned then return false end
	if winapi.enabled then
		local hPipe = winapi.C.CreateFileW(
			winapi.socket_wc,
			winapi.GENERIC_WRITE,
			0,
			nil,
			winapi.OPEN_EXISTING,
			winapi._flags,
			nil
		)
		if hPipe ~= winapi.INVALID_HANDLE_VALUE then
			local buf = command .. "\n"
			winapi.C.SetNamedPipeHandleState(hPipe, winapi.PIPE_NOWAIT, nil, nil)
			local ok = winapi.C.WriteFile(hPipe, buf, #buf, winapi._written, nil)
			winapi.C.CloseHandle(hPipe)
			return ok ~= false
		end
		return false
	end

	local cmd_line = command .. "\n"
	if os_name == "windows" then
		if state.file and state.file_bytes + #cmd_line >= 4096 then
			pcall(function() state.file:close() end)
			state.file = nil
			state.file_bytes = 0
		end
		if not state.file then
			state.file = io.open("\\\\.\\pipe\\" .. options.socket, "r+b")
		end
	elseif not state.file then
		state.file = io.open(options.socket, "r+")
	end
	if state.file then
		state.file_bytes = (state.file:seek("end") or state.file_bytes) + #cmd_line
		state.file:write(cmd_line)
		state.file:flush()
		return true
	end
	return false
end

local activity_timer
local file_timer
local seek_timer

local function reset_inactivity_timer()
	if options.quit_after_inactivity > 0 and activity_timer then
		activity_timer:kill()
		activity_timer:resume()
	end
end

local function draw(w, h)
	if not w or not h or not state.show_thumbnail or state.x == nil or state.y == nil then
		return
	end
	local cmd_x, cmd_y = state.x, state.y
	local cmd_w, cmd_h = nil, nil
	local preview = preview_draw and preview_draw.x and preview_draw.y and preview_draw.w and preview_draw.h
	if preview then
		cmd_x, cmd_y = preview_draw.x, preview_draw.y
		cmd_w, cmd_h = preview_draw.w, preview_draw.h
	end
	pcall(mp.command_native, {
		name = "overlay-add",
		id = options.overlay_id,
		x = cmd_x,
		y = cmd_y,
		file = options.tnpath .. ".bgra",
		offset = 0,
		fmt = "bgra",
		w = w,
		h = h,
		stride = 4 * w,
		dw = cmd_w,
		dh = cmd_h,
	})
	if preview then
		local osd_w, osd_h = mp.get_osd_size()
		if osd_w > 0 and osd_h > 0 then
			preview_ass.res_x = osd_w
			preview_ass.res_y = osd_h
			preview_ass.data = preview_draw.ass or ""
			preview_ass:update()
		end
	end
end

local function clear_overlay()
	if file_timer then file_timer:kill() end
	if seek_timer then seek_timer:kill() end
	state.pending_ipc_seek = false
	state.show_thumbnail = false
	state.last_x = nil
	state.last_y = nil
	preview_ass:remove()
	reset_inactivity_timer()
	mp.command_native_async({name = "overlay-remove", id = options.overlay_id}, function() end)
end

local function stop_worker()
	if activity_timer then activity_timer:kill() end
	if state.spawned then
		run_ipc("quit")
		state.spawned = false
	end
	state.real_w, state.real_h = nil, nil
	state.pending_ipc_seek = false
	clear_overlay()
	remove_thumbnail_files()
end

activity_timer = mp.add_timeout(options.quit_after_inactivity, function()
	if state.show_thumbnail then
		reset_inactivity_timer()
		return
	end
	stop_worker()
end)
activity_timer:kill()

local function spawn_worker(start_sec)
	local path = state.properties["path"]
	if not path or path == "" then return end

	remove_thumbnail_files()
	reset_inactivity_timer()

	local vid = state.properties["vid"]
	local args = {
		options.mpv_path,
		"--config=no",
		"--terminal=no",
		"--msg-level=all=no",
		"--idle=yes",
		"--keep-open=always",
		"--pause=yes",
		"--ao=null",
		"--osc=no",
		"--load-scripts=no",
		"--load-stats-overlay=no",
		"--load-console=no",
		"--load-commands=no",
		"--load-auto-profiles=no",
		"--load-select=no",
		"--load-context-menu=no",
		"--load-positioning=no",
		"--video-osd=no",
		"--autoload-files=no",
		"--vd-lavc-skiploopfilter=all",
		"--vd-lavc-skipidct=all",
		"--vd-lavc-fast",
		"--vd-lavc-threads=" .. tostring(options.sw_threads),
		"--hwdec=no",
		"--demuxer-max-bytes=2MiB",
		"--demuxer-readahead-secs=0",
		"--edition=" .. tostring(state.properties["edition"] or "auto"),
		"--vid=" .. tostring(vid or "auto"),
		"--sub=no",
		"--audio=no",
		"--start=" .. string.format("%.3f", math.max(0, start_sec)),
		"--dither-depth=no",
		"--deinterlace=no",
		"--input-media-keys=no",
		"--vf=scale=w=" .. tostring(state.effective_w) .. ":h=" .. tostring(state.effective_h) .. ",format=fmt=bgra",
		"--ovc=rawvideo",
		"--of=image2",
		"--ofopts=update=1",
		"--ocopy-metadata=no",
		"--o=" .. options.tnpath,
	}

	local headers = mp.get_property_native("http-header-fields")
	local has_referer = false
	if type(headers) == "table" then
		for _, h in ipairs(headers) do
			if type(h) == "string" and h ~= "" then
				table.insert(args, "--http-header-fields-append=" .. h)
				if h:lower():match("^referer:") then
					has_referer = true
				end
			end
		end
	end
	if not has_referer and options.referer ~= "" then
		table.insert(args, "--http-header-fields-append=Referer: " .. options.referer)
	end

	if os_name == "windows" then
		table.insert(args, "--media-controls=no")
		table.insert(args, "--input-ipc-server=" .. options.socket)
	else
		table.insert(args, "--input-ipc-server=" .. options.socket)
	end

	table.insert(args, "--")
	table.insert(args, path)

	state.spawned = true
	state.pending_ipc_seek = false

	mp.command_native_async({
		name = "subprocess",
		args = args,
		playback_only = true,
	}, function(success, result)
		state.spawned = false
		if not success or (result and result.status ~= 0 and result.status ~= -2) then
			mp.msg.warn("sjtu_thumb child mpv exited unexpectedly")
		end
	end)
end

local function send_seek()
	if not state.last_seek_time or not state.spawned then return end
	local ok = run_ipc(string.format("async seek %.3f absolute+keyframes", math.max(0, state.last_seek_time)))
	state.pending_ipc_seek = not ok
end

seek_timer = mp.add_periodic_timer(options.frequency, function()
	if not state.show_thumbnail then
		seek_timer:kill()
		return
	end
	if state.pending_ipc_seek then
		send_seek()
	end
end)
seek_timer:kill()

file_timer = mp.add_periodic_timer(1 / 60, function()
	if state.pending_ipc_seek and state.spawned then
		send_seek()
	end
	local w, h = check_new_thumb()
	if w and h then
		state.real_w, state.real_h = w, h
		if state.real_w ~= state.last_real_w or state.real_h ~= state.last_real_h then
			state.last_real_w, state.last_real_h = state.real_w, state.real_h
			broadcast_info(state.real_w, state.real_h)
		end
		if not state.show_thumbnail then
			file_timer:kill()
		end
		draw(state.real_w, state.real_h)
	end
end)
file_timer:kill()

local function thumb(time_sec, r_x, r_y)
	if state.disabled then return end
	local t = tonumber(time_sec)
	if not t then return end
	t = math.max(0, t)

	local nx = tonumber(r_x)
	local ny = tonumber(r_y)
	if nx and ny then
		state.x = math.floor(nx + 0.5)
		state.y = math.floor(ny + 0.5)
	else
		state.x, state.y = nil, nil
	end

	local was_showing = state.show_thumbnail
	if state.last_x ~= state.x or state.last_y ~= state.y or not state.show_thumbnail then
		state.show_thumbnail = true
		state.last_x, state.last_y = state.x, state.y
		if was_showing and state.real_w and state.real_h then
			draw(state.real_w, state.real_h)
		end
	end

	reset_inactivity_timer()

	if state.last_seek_time and math.abs(t - state.last_seek_time) < options.min_seek_delta then
		if not was_showing and state.real_w and state.real_h then
			draw(state.real_w, state.real_h)
		end
		return
	end

	state.last_seek_time = t
	if not state.spawned then
		spawn_worker(t)
	else
		send_seek()
	end
	if not seek_timer:is_enabled() then seek_timer:resume() end
	if not file_timer:is_enabled() then file_timer:resume() end
end

mp.observe_property("user-data/osc/draw-preview", "native", function(_, value)
	preview_draw = value
	if preview_draw == nil then
		clear_overlay()
		return
	end
	local hover_sec = mp.get_property_number("user-data/osc/hover-sec")
	thumb(hover_sec, value.x, value.y)
end)

mp.register_script_message("thumb", thumb)
mp.register_script_message("clear", clear_overlay)

mp.observe_property("current-tracks/video", "native", function(name, value)
	state.properties[name] = value
end)
mp.observe_property("track-list", "native", function(_, value)
	if type(value) ~= "table" then return end
	for _, track in ipairs(value) do
		if track.type == "video" and track.selected then
			state.properties["current-tracks/video"] = track
			return
		end
	end
end)
mp.observe_property("display-hidpi-scale", "native", function(name, value)
	state.properties[name] = value
	state.dirty = true
end)
mp.observe_property("video-params", "native", function(name, value)
	state.properties[name] = value
	state.dirty = true
end)
mp.observe_property("video-dec-params", "native", function(name, value)
	state.properties[name] = value
	state.dirty = true
end)
mp.observe_property("path", "native", function(name, value)
	state.properties[name] = value
end)
mp.observe_property("edition", "native", function(name, value)
	state.properties[name] = value
end)
mp.observe_property("vid", "native", function(name, value)
	state.properties[name] = value
	if type(value) == "boolean" and value == false then
		state.has_vid = 0
		state.last_has_vid = 0
		broadcast_info(state.effective_w, state.effective_h)
		clear_overlay()
		return
	end
	state.has_vid = (value and value ~= "no" and value ~= 0) and 1 or 0
	if state.spawned then
		run_ipc("set vid " .. tostring(value))
		state.dirty = true
	end
end)

mp.register_idle(function()
	if not state.dirty or not state.properties["video-params"] then return end
	state.dirty = false
	local old_w, old_h = state.effective_w, state.effective_h
	calc_dimensions()
	local resized = (old_w ~= state.effective_w) or (old_h ~= state.effective_h)
	if resized or (state.last_has_vid ~= state.has_vid and state.has_vid ~= 0) then
		broadcast_info(state.effective_w, state.effective_h)
	end
	if state.spawned and resized then
		local seek_time = state.last_seek_time or mp.get_property_number("time-pos", 0)
		stop_worker()
		spawn_worker(seek_time)
		if file_timer then file_timer:resume() end
	end
	state.last_has_vid = state.has_vid
end)

mp.register_event("file-loaded", function()
	preview_draw = nil
	stop_worker()
	state.last_seek_time = nil
	calc_dimensions()
	broadcast_info(state.effective_w, state.effective_h)
end)

mp.register_event("shutdown", stop_worker)
