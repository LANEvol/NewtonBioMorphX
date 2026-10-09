//
// Created by jahanbak on 05/10/2026.
//

#ifndef NEWTONBIOMORPHX_MAIN_H
#define NEWTONBIOMORPHX_MAIN_H

#include "imgui.h"
#include "misc/cpp/imgui_stdlib.h"
#include <string>

namespace LagSol {

// Insert '\n' so that no line is wider than `width`. Input has no '\n'.
// Every character is kept, so removing the '\n' restores the original.
std::string wrapAt(const std::string& s, float width)
{
    std::string out;
    out.reserve(s.size() + 8);
    ImFont* font = ImGui::GetFont();
    const float scale = ImGui::GetFontSize() / font->FontSize;
    const char* p   = s.c_str();
    const char* end = p + s.size();
    while (p < end) {
        const char* q = font->CalcWordWrapPositionA(scale, p, end, width);
        if (q <= p) q = p + 1;                  // always make progress
        out.append(p, q);
        p = q;
        if (p < end) out.push_back('\n');
    }
    return out;
}

std::string stripNewlines(const char* s, int n)
{
    std::string out;
    out.reserve(n);
    for (int i = 0; i < n; ++i)
        if (s[i] != '\n' && s[i] != '\r') out.push_back(s[i]);
    return out;
}

// Re-wraps while typing and keeps cursor/selection on the same real characters.
int wrapCallback(ImGuiInputTextCallbackData* d)
{
    const float width = *static_cast<float*>(d->UserData);

    auto realCount = [&](int upto) {
        int n = 0;
        for (int i = 0; i < upto; ++i)
            if (d->Buf[i] != '\n' && d->Buf[i] != '\r') ++n;
        return n;
    };
    const int cur  = realCount(d->CursorPos);
    const int selS = realCount(d->SelectionStart);
    const int selE = realCount(d->SelectionEnd);

    const std::string current(d->Buf, d->BufTextLen);
    const std::string wrapped = wrapAt(stripNewlines(d->Buf, d->BufTextLen), width);
    if (wrapped == current) return 0;

    d->DeleteChars(0, d->BufTextLen);
    d->InsertChars(0, wrapped.c_str());

    auto toIndex = [&](int real) {
        int n = 0, i = 0;
        while (i < (int)wrapped.size() && n < real) {
            if (wrapped[i] != '\n') ++n;
            ++i;
        }
        return i;
    };
    d->CursorPos      = toIndex(cur);
    d->SelectionStart = toIndex(selS);
    d->SelectionEnd   = toIndex(selE);
    return 0;
}

// Word-wrapping text field. Returns true when `text` changed.
// outerWidth <= 0 means "use the remaining width of the current row/cell".
inline bool InputTextAutoWrap(const char* id, std::string& text, float outerWidth = 0.0f)
{
    const ImGuiStyle& st = ImGui::GetStyle();
    if (outerWidth <= 0.0f) outerWidth = ImGui::GetContentRegionAvail().x;
    float inner = outerWidth - 2.0f * st.FramePadding.x - 2.0f;   // small safety margin

    std::string shown = wrapAt(stripNewlines(text.c_str(), (int)text.size()), inner);
    int lines = 1;
    for (char c : shown) if (c == '\n') ++lines;

    const float h = lines * ImGui::GetTextLineHeight() + 2.0f * st.FramePadding.y;

    const bool changed = ImGui::InputTextMultiline(
        id, &shown, ImVec2(outerWidth, h),
        ImGuiInputTextFlags_NoHorizontalScroll | ImGuiInputTextFlags_CallbackEdit,
        wrapCallback, &inner);

    if (changed)
        text = stripNewlines(shown.c_str(), (int)shown.size());
    return changed;
}

    // Small circled "i"; shows a wrapped tooltip while the cursor hovers it.
inline void InfoMarker(const std::string& text) {
    const float r = ImGui::GetFontSize() * 0.5f;
    const ImVec2 pos = ImGui::GetCursorScreenPos();

    ImGui::InvisibleButton("##info", ImVec2(2.0f * r, 2.0f * r));
    const bool hovered = ImGui::IsItemHovered();

    const ImVec2 c(pos.x + r, pos.y + r);
    const ImU32 col = ImGui::GetColorU32(hovered ? ImGuiCol_Text : ImGuiCol_TextDisabled);
    ImDrawList* dl = ImGui::GetWindowDrawList();
    dl->AddCircle(c, r, col, 0, 1.5f);

    const ImVec2 ts = ImGui::CalcTextSize("i");
    dl->AddText(ImVec2(c.x - ts.x * 0.5f, c.y - ts.y * 0.5f), col, "i");

    if (hovered) {
        ImGui::BeginTooltip();
        ImGui::PushTextWrapPos(ImGui::GetFontSize() * 25.0f);   // tooltip width ~25 characters
        ImGui::TextUnformatted(text.c_str());
        ImGui::PopTextWrapPos();
        ImGui::EndTooltip();
    }
}

} // namespace

#endif //NEWTONBIOMORPHX_MAIN_H
