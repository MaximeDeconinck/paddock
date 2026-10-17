# paddock demo: Claude Code delegates private work to local models.
# Run through demo/record.sh: it stages ~/releves, cds there and renders the result.

Output ".raw.mp4"

Require claude
Require paddock

Set Shell zsh
Set FontSize 18
# Window content size from demo/frame.py (render.sh draws the window around it).
Set Width 1620
Set Height 888
Set Padding 22
Set Theme "Catppuccin Mocha"
Set TypingSpeed 35ms

# Setup, off camera: isolated Claude Code (project settings only, paddock as the sole MCP server).
Hide
Type "export PS1='%F{cyan}~/releves%f $ ' && alias claude='command claude --setting-sources project --strict-mcp-config --mcp-config .mcp-demo.json --permission-mode default' && clear"
Enter
Sleep 1s
# First launch in a new folder asks for workspace trust: accept it off camera
# (a no-op once trusted), then quit.
Type "claude"
Enter
Sleep 6s
Down
Enter
Sleep 6s
Ctrl+C
Sleep 300ms
Ctrl+C
Sleep 1s
Type "clear"
Enter
Sleep 500ms
Show

Type "ls data/"
Sleep 400ms
Enter
Sleep 1.5s
Type "claude"
Sleep 400ms
Enter
Sleep 6s

# Pasted, not typed.
Type@1ms "Utilise paddock pour analyser mes relevés de septembre (data/carte.csv, data/compte-courant.csv, data/compte-joint.csv) avec le petit modèle local Qwen3 4B : un sous-agent par fichier, en parallèle, sans jamais lire les fichiers toi-même. Vas-y directement, puis fais-moi le bilan : où part mon argent et ce qui cloche."
Sleep 800ms
Enter

# Let Claude work. No reliable "done" marker on screen: wait long enough,
# render.sh compresses the idle stretches.
Sleep 7m

# Quit off camera: the last recorded frame is the final answer.
Hide
Ctrl+C
Sleep 300ms
Ctrl+C
Sleep 1s
