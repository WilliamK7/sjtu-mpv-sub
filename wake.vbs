Set fso = CreateObject("Scripting.FileSystemObject")
Set shell = CreateObject("WScript.Shell")

rootDir = fso.GetParentFolderName(WScript.ScriptFullName)
pythonw = fso.BuildPath(rootDir, ".venv\Scripts\pythonw.exe")

shell.CurrentDirectory = rootDir
cmd = """" & pythonw & """ -m sjtu_mpv_sub.cli serve --ephemeral"
shell.Run cmd, 0, False
