#pragma once

#include <cstdio>
#include <ostream>
#include <string>
#include <vector>

namespace bq::tui {

enum class Color {
    Default,
    Black,
    Red,
    Green,
    Yellow,
    Blue,
    Magenta,
    Cyan,
    White,
    BrightBlack,
    BrightRed,
    BrightGreen,
    BrightYellow,
    BrightBlue,
    BrightMagenta,
    BrightCyan,
    BrightWhite,
};

struct Style {
    Color fg = Color::Default;
    Color bg = Color::Default;
    bool bold = false;
    bool dim = false;
    bool inverse = false;
};

struct TerminalCapabilities {
    bool is_tty = false;
    bool ansi = false;
    bool color = false;
    bool truecolor = false;
    bool dumb = false;
    int columns = 80;
    int rows = 24;
};

struct BoxOptions {
    std::string title;
    int width = 0;
    int padding = 1;
    bool wrap = true;
    Style border_style = { Color::BrightBlack, Color::Default, false, false, false };
    Style title_style = { Color::BrightCyan, Color::Default, true, false, false };
};

struct ProductInfo {
    std::string name = "advanced-gguf-quantizer";
    std::string version = "dev";
    std::string subtitle = "GGUF NVFP4 / MXFP8 / experimental MXFP6 quantization shell";
};

struct StatusItem {
    std::string label;
    std::string value;
    std::string detail;
    Style value_style = { Color::BrightWhite, Color::Default, false, false, false };
};

struct MenuOption {
    std::string label;
    std::string description;
    std::string command;
    bool disabled = false;
};

struct SlashCommand {
    std::string name;
    std::string args;
    std::string description;
};

struct InputPrompt {
    std::string label;
    std::string value;
    std::string placeholder = "enter value";
    std::string hint;
    bool required = false;
    Style value_style = { Color::BrightWhite, Color::Default, false, false, false };
};

TerminalCapabilities detect_terminal(FILE * stream = stdout);
bool should_use_color(const TerminalCapabilities & caps);

Style accent();
Style success();
Style warning();
Style error();
Style muted();
Style selected();
Style normal();

std::string ansi(const Style & style, const TerminalCapabilities & caps);
std::string reset(const TerminalCapabilities & caps);
std::string paint(std::string text, const Style & style, const TerminalCapabilities & caps);
std::string strip_ansi(const std::string & text);

size_t display_width(const std::string & text);
std::vector<std::string> wrap_words(const std::string & text, size_t width);
std::string truncate_to_width(const std::string & text, size_t width);

std::string horizontal_rule(size_t width, char fill = '-');
std::string render_box(const std::vector<std::string> & lines, const BoxOptions & options, const TerminalCapabilities & caps);
std::string render_section_header(
        const std::string & title,
        const std::string & subtitle,
        const TerminalCapabilities & caps,
        size_t width = 0);
std::string render_input_prompt(const InputPrompt & prompt, const TerminalCapabilities & caps);
std::string render_branded_header(const ProductInfo & product, const TerminalCapabilities & caps);
std::string render_status_panel(
        const ProductInfo & product,
        const std::vector<StatusItem> & items,
        const TerminalCapabilities & caps);

std::vector<SlashCommand> default_slash_commands();
std::string render_slash_help(const std::vector<SlashCommand> & commands, const TerminalCapabilities & caps);

std::string format_menu_option(
        size_t index,
        const MenuOption & option,
        bool is_selected,
        const TerminalCapabilities & caps,
        size_t width = 0);
std::string render_menu(
        const std::string & title,
        const std::vector<MenuOption> & options,
        size_t selected_index,
        const TerminalCapabilities & caps,
        size_t max_visible_rows = 0);

void print(std::ostream & out, const std::string & rendered);

} // namespace bq::tui
