# Demo of the paddock TUI for the README.
# Regenerate after UI changes: cargo build --release && vhs docs/demo.tape

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

# Open the detail popup of the selected model.
Enter
Sleep 3s
Escape
Sleep 1s

# Search.
Type "/"
Sleep 500ms
Type@120ms "qwen"
Sleep 500ms
Enter
Sleep 2s

# Switch the scoring to the coding use-case.
Type "c"
Sleep 2.5s

# Back to general, quit.
Type "g"
Sleep 1.5s
Type "q"
Sleep 1s
