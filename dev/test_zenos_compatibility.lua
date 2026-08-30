#!/usr/bin/env luajit
--[[--
Unit tests for Audiobook plugin ZenOS compatibility hooks.
Run from repo root:
  luajit dev/test_zenos_compatibility.lua
--]]

local function assert_eq(expected, actual, msg)
    if expected ~= actual then
        error(string.format("Assertion failed: expected %s, got %s. (%s)",
            tostring(expected), tostring(actual), msg or ""), 2)
    end
end

local function assert_true(val, msg)
    if not val then
        error("Assertion failed: expected true, got " .. tostring(val) .. ". (" .. (msg or "") .. ")", 2)
    end
end

print("── Testing Audiobook plugin ZenOS compatibility ──")

-- 1. Mock minimal environment
_G._ = function(s) return s end
_G.T = function(fmt, ...) return fmt end
_G.logger = {
    dbg = function() end,
    warn = function() end,
    info = function() end,
    err = function() end,
}
_G.UIManager = {
    close = function() end,
    scheduleIn = function(_, fn) fn() end,
    show = function() end,
}
_G.Device = {
    screen = { getWidth = function() return 600 end, getHeight = function() return 800 end },
    isKindle = function() return false end,
}

-- Mock Audiobook plugin table
local Audiobook = {}
Audiobook.__index = Audiobook

function Audiobook:new(o)
    o = o or {}
    setmetatable(o, self)
    return o
end

function Audiobook:_getActiveInstance()
    local ok_ru, ReaderUI = pcall(require, "apps/reader/readerui")
    if ok_ru and ReaderUI and ReaderUI.instance and ReaderUI.instance.audiobook then
        return ReaderUI.instance.audiobook
    end
    local ok_fm, FileManager = pcall(require, "apps/filemanager/filemanager")
    if ok_fm and FileManager and FileManager.instance and FileManager.instance.audiobook then
        return FileManager.instance.audiobook
    end
    return self
end

function Audiobook:_hasActiveDocument()
    local active = self:_getActiveInstance()
    if active and active.ui and active.ui.document then return true end
    local ok_ru, ReaderUI = pcall(require, "apps/reader/readerui")
    if ok_ru and ReaderUI and ReaderUI.instance and ReaderUI.instance.document then
        return true
    end
    return false
end

function Audiobook:_hasMediaOverlays()
    return true
end

function Audiobook:_installZenOSCompatibility()
    local plugin = self

    -- 1. Extend ZenOS LookupPluginItems if available
    local function patchLookupPluginItems(LookupPluginItems)
        if not LookupPluginItems or LookupPluginItems._audiobook_patched then return end
        LookupPluginItems._audiobook_patched = true

        local orig_settingForHighlightKey = LookupPluginItems.settingForHighlightKey
        LookupPluginItems.settingForHighlightKey = function(key)
            if type(key) == "string" then
                local name = key:match("^%d+_(.*)$") or key
                if name == "play_aligned" or name:find("audiobook", 1, true) then
                    return "show_audiobook"
                end
            end
            if orig_settingForHighlightKey then
                return orig_settingForHighlightKey(key)
            end
        end

        local orig_settingForDictButton = LookupPluginItems.settingForDictButton
        LookupPluginItems.settingForDictButton = function(button)
            if type(button) == "table" then
                local id = button.id
                if id == "audiobook_play_aligned" or (type(id) == "string" and id:find("audiobook", 1, true)) then
                    return "show_audiobook"
                end
                local label = button.text
                if type(label) ~= "string" and type(button.text_func) == "function" then
                    local ok, text = pcall(button.text_func)
                    if ok then label = text end
                end
                if type(label) == "string" and (label:find("audiobook", 1, true) or label:find("Read aloud", 1, true)) then
                    return "show_audiobook"
                end
            end
            if orig_settingForDictButton then
                return orig_settingForDictButton(button)
            end
        end

        local orig_shouldShow = LookupPluginItems.shouldShow
        LookupPluginItems.shouldShow = function(config, setting)
            if setting == "show_audiobook" then
                local lookup = type(config) == "table" and config.highlight_lookup
                if type(lookup) == "table" and lookup.show_audiobook == false then
                    return false
                end
                return true
            end
            if orig_shouldShow then
                return orig_shouldShow(config, setting)
            end
            return true
        end
    end

    local lpi = package.loaded["modules/reader/lookup_plugin_items"]
    if lpi then
        patchLookupPluginItems(lpi)
    else
        local ok_lpi, loaded_lpi = pcall(require, "modules/reader/lookup_plugin_items")
        if ok_lpi and loaded_lpi then
            patchLookupPluginItems(loaded_lpi)
        end
    end

    -- 2. Fallback patch on DictQuickLookup in case ZenOS DictQuickLookup patch is active
    local ok_dql, DictQuickLookup = pcall(require, "ui/widget/dictquicklookup")
    if ok_dql and DictQuickLookup and DictQuickLookup.buildButtonLayout and not DictQuickLookup._audiobook_fallback_patched then
        DictQuickLookup._audiobook_fallback_patched = true
        local orig_buildButtonLayout = DictQuickLookup.buildButtonLayout
        DictQuickLookup.buildButtonLayout = function(self_dql)
            local layout = orig_buildButtonLayout(self_dql)
            if not layout or self_dql.is_wiki or self_dql.is_wiki_fullpage then
                return layout
            end

            local active = plugin:_getActiveInstance()
            if not (active and active._init_ok and active:_hasMediaOverlays()) then
                return layout
            end

            for _i, row in ipairs(layout) do
                for _j, btn in ipairs(row) do
                    if btn.id == "audiobook_play_aligned" then
                        return layout
                    end
                end
            end

            local play_btn = {
                id = "audiobook_play_aligned",
                text = _("Play aligned audiobook from here"),
                font_bold = false,
                callback = function()
                    local selected_text = nil
                    if self_dql.highlight and self_dql.highlight.selected_text then
                        selected_text = self_dql.highlight.selected_text
                    end
                    UIManager:close(self_dql)
                    UIManager:scheduleIn(0.3, function()
                        active:startAlignedAudioFromSelection(selected_text)
                    end)
                end,
            }
            table.insert(layout, { play_btn })
            return layout
        end
    end

    -- 3. Fallback patch on ReaderHighlight in case ZenOS HighlightMenu patch is active
    local ok_rh, ReaderHighlight = pcall(require, "apps/reader/modules/readerhighlight")
    if ok_rh and ReaderHighlight and ReaderHighlight.onShowHighlightMenu and not ReaderHighlight._audiobook_fallback_patched then
        ReaderHighlight._audiobook_fallback_patched = true
        local orig_onShowHighlightMenu = ReaderHighlight.onShowHighlightMenu
        ReaderHighlight.onShowHighlightMenu = function(self_rh, index)
            local active = plugin:_getActiveInstance()
            local had_dlg = self_rh.highlight_dialog

            orig_onShowHighlightMenu(self_rh, index)

            if not (active and active._init_ok and active:_hasMediaOverlays()) then
                return
            end

            local dlg = self_rh.highlight_dialog
            if not dlg or dlg == had_dlg or not dlg.buttons then
                return
            end

            local found = false
            for _i, row in ipairs(dlg.buttons) do
                for _j, btn in ipairs(row) do
                    if btn.text == _("Play aligned audiobook from here") or btn.id == "audiobook_play_aligned" then
                        found = true
                        break
                    end
                end
                if found then break end
            end

            if not found then
                local play_btn = {
                    text = _("Play aligned audiobook from here"),
                    callback = function()
                        local selected_text = self_rh.selected_text
                        if self_rh.onClose then self_rh:onClose() end
                        if dlg.onClose then dlg:onClose() end
                        UIManager:scheduleIn(0.3, function()
                            active:startAlignedAudioFromSelection(selected_text)
                        end)
                    end,
                }
                table.insert(dlg.buttons, { play_btn })
            end
        end
    end
end

function Audiobook:addToMainMenu(menu_items)
    local active = self:_getActiveInstance()
    if active ~= self and active.addToMainMenu then
        return active:addToMainMenu(menu_items)
    end
    menu_items.audiobook = {
        text = _("Audiobook Read-Along"),
        sub_item_table = {
            {
                text = _("Start Text-to-Speech from current page"),
                enabled_func = function() return (self.ui and self.ui.document) or false end,
                callback = function() end,
            },
        },
    }
end

-- Test 1: LookupPluginItems monkey-patching
print("Test 1: LookupPluginItems integration");
(function()
    local mock_lpi = {
        settingForHighlightKey = function(k) if k == "xray_lookup" then return "show_xray" end end,
        settingForDictButton = function(b) if b.id == "xray_lookup" then return "show_xray" end end,
        shouldShow = function(cfg, setting)
            if setting == "show_xray" then return true end
            return false
        end,
    }
    package.loaded["modules/reader/lookup_plugin_items"] = mock_lpi

    local plugin = Audiobook:new{ _init_ok = true }
    plugin:_installZenOSCompatibility()

    assert_eq("show_audiobook", mock_lpi.settingForHighlightKey("16_play_aligned"), "highlight key mapping")
    assert_eq("show_audiobook", mock_lpi.settingForDictButton({ id = "audiobook_play_aligned" }), "dict button id mapping")
    assert_eq("show_audiobook", mock_lpi.settingForDictButton({ text = "Play aligned audiobook from here" }), "dict button text mapping")
    assert_true(mock_lpi.shouldShow({}, "show_audiobook"), "shouldShow default true")
    assert_eq(false, mock_lpi.shouldShow({ highlight_lookup = { show_audiobook = false } }, "show_audiobook"), "shouldShow respects false")
    print("  ✓ LookupPluginItems hooks passed")
end)()

-- Test 2: Active instance resolution when ReaderUI is active
print("Test 2: Active ReaderUI resolution from docless instance");
(function()
    local live_reader_audiobook = Audiobook:new{
        _init_ok = true,
        ui = { document = { file = "test.epub" } },
    }
    package.loaded["apps/reader/readerui"] = {
        instance = {
            document = live_reader_audiobook.ui.document,
            audiobook = live_reader_audiobook,
        },
    }

    local docless_loader_instance = Audiobook:new{
        _init_ok = true,
        ui = nil,
    }

    assert_eq(live_reader_audiobook, docless_loader_instance:_getActiveInstance(), "getActiveInstance resolves ReaderUI audiobook")
    assert_true(docless_loader_instance:_hasActiveDocument(), "hasActiveDocument returns true")

    local menu_items = {}
    docless_loader_instance:addToMainMenu(menu_items)
    assert_true(menu_items.audiobook ~= nil, "audiobook menu created")
    local tts_item = menu_items.audiobook.sub_item_table[1]
    assert_true(tts_item.enabled_func(), "TTS item is enabled on active instance")
    print("  ✓ Active instance resolution passed")
end)()

-- Test 3: DictQuickLookup fallback button injection
print("Test 3: DictQuickLookup fallback injection");
(function()
    local mock_dql = {
        buildButtonLayout = function(self_dql)
            -- ZenOS filtered out third party buttons, returns only core
            return {
                { { id = "highlight" }, { id = "search" } }
            }
        end,
    }
    package.loaded["ui/widget/dictquicklookup"] = mock_dql

    local plugin = Audiobook:new{ _init_ok = true }
    plugin:_installZenOSCompatibility()

    local layout = mock_dql.buildButtonLayout({ highlight = { selected_text = "test" } })
    assert_eq(2, #layout, "audiobook button row was appended to filtered layout")
    assert_eq("audiobook_play_aligned", layout[2][1].id, "appended button has audiobook_play_aligned id")
    print("  ✓ DictQuickLookup fallback passed")
end)()

-- Test 4: ReaderHighlight fallback button injection
print("Test 4: ReaderHighlight fallback injection");
(function()
    local mock_rh = {
        onShowHighlightMenu = function(self_rh, index)
            -- ZenOS built custom dialog without audiobook button
            self_rh.highlight_dialog = {
                buttons = {
                    { { text = "Highlight" }, { text = "Search" } }
                }
            }
        end,
    }
    package.loaded["apps/reader/modules/readerhighlight"] = mock_rh

    local plugin = Audiobook:new{ _init_ok = true }
    plugin:_installZenOSCompatibility()

    local rh_instance = { selected_text = "Sample sentence" }
    mock_rh.onShowHighlightMenu(rh_instance, 1)

    assert_eq(2, #rh_instance.highlight_dialog.buttons, "audiobook button row was appended to highlight dialog")
    assert_eq("Play aligned audiobook from here", rh_instance.highlight_dialog.buttons[2][1].text, "appended button has correct text")
    print("  ✓ ReaderHighlight fallback passed")
end)()

print("── ALL ZENOS COMPATIBILITY TESTS PASSED! ──")
