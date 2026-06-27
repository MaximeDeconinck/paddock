# Demo of the paddock TUI for the README.
# Regenerate after UI changes: cargo build --release && vhs docs/demo.tape
# A running server + installed Ollama models make the servers tab non-empty;
# seed one first, e.g. `paddock serve <cached-model>`.

Output docs/demo.gif

Set FontSize 13
Set Width 1280
Set Height 720
Set Padding 12
Set TypingSpeed 60ms
Set Theme "Catppuccin Mocha"

# Put the release binary on PATH without showing the plumbing.
Hide
Type "export PATH=$PWD/target/release:$PATH && clear"
Enter
Show

Type "paddock"
Sleep 500ms
Enter
Sleep 2.5s

# Browse the ranking.
Down@250ms 5
Sleep 1.5s

# Search for a model with several quantizations.
Type "/"
Sleep 500ms
Type@120ms "qwen"
Sleep 500ms
Enter
Sleep 1.5s

# Open the detail popup and pick a smaller quant (the quant picker): the table
# lists every quant with its memory / tok-s / fit; arrows move the selection.
Enter
Sleep 2.5s
Down@500ms 3
Sleep 2.5s
Escape
Sleep 1s

# Re-score for the coding use-case, then back to general.
Type "c"
Sleep 2s
Type "g"
Sleep 1.5s

# Switch to the servers tab: running endpoints on top, the AVAILABLE group
# below (greyed, relaunchable) - Ollama models + previously served catalog entries.
Tab
Sleep 3s
Down@300ms 4
Sleep 2.5s

# Back to the models tab, quit.
Tab
Sleep 1s
Type "q"
Sleep 1s
