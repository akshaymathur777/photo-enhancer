; Photo Enhancer — Inno Setup installer script
; Requires Inno Setup 6+: https://jrsoftware.org/isinfo.php
; Build the app first: build.bat

#define MyAppName "Photo Enhancer"
#define MyAppVersion "0.1.8"
#define MyAppPublisher "Photo Enhancer"
#define MyAppExeName "PhotoEnhancer.exe"
#define BuildDir "dist\PhotoEnhancer"

[Setup]
AppId={{A7B3C4D5-E6F7-4890-ABCD-EF1234567890}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
DefaultDirName={autopf}\{#MyAppName}
DefaultGroupName={#MyAppName}
DisableProgramGroupPage=yes
OutputDir=dist
OutputBaseFilename=PhotoEnhancer-Setup
Compression=lzma2/ultra64
SolidCompression=yes
WizardStyle=modern
PrivilegesRequired=lowest
ArchitecturesInstallIn64BitMode=x64compatible
UninstallDisplayIcon={app}\{#MyAppExeName}

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
Source: "{#BuildDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs
; NOTE: Do not use "Flags: ignoreversion" on any shared system files

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "{cm:LaunchProgram,{#StringChange(MyAppName, '&', '&&')}}"; Flags: nowait postinstall skipifsilent

[UninstallDelete]
Type: filesandordirs; Name: "{userappdata}\PhotoEnhancer"

[Code]
function InitializeSetup(): Boolean;
begin
  if not FileExists(ExpandConstant('{src}\{#BuildDir}\{#MyAppExeName}')) then
  begin
    MsgBox('Build output not found. Run build.bat first.', mbError, MB_OK);
    Result := False;
  end
  else
    Result := True;
end;
