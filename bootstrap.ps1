<#
  bootstrap.ps1 -- fetch everything a clone needs and place it correctly.

  Downloads the Blender builds and test models from official blender.org URLs into
  blender/<version>/ and models/. Nothing large lives in git -- the repo is code +
  this script + the config. Idempotent: existing items are skipped (use -Force to
  re-download). Uses built-in curl.exe + tar.exe (Windows 10 1803+/11).

  Run:  double-click bootstrap.cmd   (or)   powershell -ExecutionPolicy Bypass -File bootstrap.ps1
#>
param([switch]$Force)
$ErrorActionPreference = "Stop"
$ROOT = Split-Path -Parent $MyInvocation.MyCommand.Path
$TMP  = Join-Path $env:TEMP "bbb_dl"
New-Item -ItemType Directory -Force $ROOT\blender,$ROOT\models,$ROOT\models\eevee,$ROOT\results,$ROOT\logs,$TMP | Out-Null

# ---- Blender builds: version -> official portable zip -----------------------
$BLENDER = @(
  @{ ver="3.3"; url="https://download.blender.org/release/Blender3.3/blender-3.3.9-windows-x64.zip" }
  @{ ver="3.6"; url="https://download.blender.org/release/Blender3.6/blender-3.6.9-windows-x64.zip" }
  @{ ver="4.2"; url="https://download.blender.org/release/Blender4.2/blender-4.2.9-windows-x64.zip" }
  @{ ver="4.5"; url="https://download.blender.org/release/Blender4.5/blender-4.5.9-windows-x64.zip" }
  @{ ver="5.2"; url="https://download.blender.org/release/Blender5.2/blender-5.2.1-windows-x64.zip" }
)

# ---- Models: name -> url, kind ('blend' = direct file, 'zip' = extract folder)
#   Splash scenes (Cycles / version-matched) and EEVEE demo scenes (EEVEE-only).
$MODELS = @(
  # --- splash (Cycles) ---  (download.blender.org file server, NOT www.blender.org)
  @{ name="blender-3.3-splash"; kind="blend"; url="https://download.blender.org/demo/splash/blender-3.3-splash.blend" }
  @{ name="blender-3.4-splash"; kind="zip";   url="https://download.blender.org/demo/splash/blender-3.4-splash.zip" }
  @{ name="blender-3.6-splash"; kind="zip";   url="https://download.blender.org/demo/splash/blender-3.6-splash.zip" }
  @{ name="blender-4.1-splash"; kind="blend"; url="https://download.blender.org/demo/splash/blender-4.1-splash.blend" }
  @{ name="blender-4.2-splash"; kind="zip";   url="https://download.blender.org/demo/splash/blender-4.2-splash.zip" }
  @{ name="blender-4.5-splash"; kind="blend"; url="https://download.blender.org/demo/splash/blender-4.5-splash.blend" }
  @{ name="blender-5.2-splash"; kind="blend"; url="https://download.blender.org/demo/splash/blender-5.2-splash.blend" }
  # --- EEVEE demo scenes ---
  @{ name="eevee/hatching";     kind="blend"; url="https://download.blender.org/demo-files/archives/art-gallery/eevee-samples/hatching-shader/eevee-hatching-by-ocean-quigley-90115ae49eb147059d9740a86c3d0966.blend" }
  @{ name="eevee/temple";       kind="blend"; url="https://download.blender.org/demo/eevee/temple/temple.blend" }
  @{ name="eevee/mr_elephant";  kind="blend"; url="https://download.blender.org/demo/eevee/mr_elephant/mr_elephant.blend" }
)

function Get-File($url,$out,$kind="any"){
  Write-Host "    GET $url"
  & curl.exe -L --fail --retry 2 -o $out $url
  if($LASTEXITCODE -ne 0){ throw "download failed ($LASTEXITCODE)" }
  # Validate the payload: many blender.org 'www' URLs return an HTML landing page
  # (not the file). Reject it instead of silently saving HTML as a .blend/.zip.
  $fs=[IO.File]::OpenRead($out); $b=New-Object byte[] 4; [void]$fs.Read($b,0,4); $fs.Close()
  $isHtml = ($b[0] -eq 0x3C -or $b[0] -eq 0x0A -or $b[0] -eq 0xEF)   # '<' / leading NL / BOM
  $isZip  = ($b[0] -eq 0x50 -and $b[1] -eq 0x4B)                     # 'PK'
  if($kind -eq "zip"   -and -not $isZip){ Remove-Item $out -Force -EA SilentlyContinue; throw "URL did not return a ZIP (looks like HTML) -- fix the URL" }
  if($kind -eq "blend" -and $isHtml){     Remove-Item $out -Force -EA SilentlyContinue; throw "URL did not return a .blend (looks like HTML) -- fix the URL" }
}

$warn = @()

Write-Host "== Blender builds ==" -ForegroundColor Cyan
foreach($b in $BLENDER){
  $dst = Join-Path "$ROOT\blender" $b.ver
  if(!$Force -and (Test-Path (Join-Path $dst "blender.exe"))){ Write-Host "  $($b.ver)  already present"; continue }
  try {
    $zip = Join-Path $TMP ("blender-"+$b.ver+".zip")
    Get-File $b.url $zip "zip"
    $ex = Join-Path $TMP ("x-"+$b.ver); Remove-Item $ex -Recurse -Force -EA SilentlyContinue; New-Item -ItemType Directory -Force $ex | Out-Null
    & tar.exe -xf $zip -C $ex; if($LASTEXITCODE -ne 0){ throw "unzip failed" }
    $child = Get-ChildItem $ex -Directory | Select-Object -First 1   # blender-x.y.z-windows-x64
    if(-not $child){ throw "archive had no top-level folder" }
    Remove-Item $dst -Recurse -Force -EA SilentlyContinue
    Move-Item $child.FullName $dst
    Remove-Item $zip,$ex -Recurse -Force -EA SilentlyContinue
    Write-Host "  $($b.ver)  OK -> blender/$($b.ver)/blender.exe" -ForegroundColor Green
  } catch { $warn += "Blender $($b.ver): $_"; Write-Warning "  $($b.ver): $_" }
}

Write-Host "== Models ==" -ForegroundColor Cyan
foreach($m in $MODELS){
  try {
    if($m.kind -eq "blend"){
      $dst = Join-Path "$ROOT\models" ($m.name + ".blend")
      New-Item -ItemType Directory -Force (Split-Path $dst) | Out-Null
      if(!$Force -and (Test-Path $dst)){ Write-Host "  $($m.name).blend  already present"; continue }
      Get-File $m.url $dst "blend"
      Write-Host "  $($m.name).blend  OK" -ForegroundColor Green
    } else {
      $dst = Join-Path "$ROOT\models" $m.name
      if(!$Force -and (Test-Path $dst)){ Write-Host "  $($m.name)/  already present"; continue }
      $zip = Join-Path $TMP ("m-"+($m.name -replace '[\\/]','_')+".zip")
      Get-File $m.url $zip "zip"
      New-Item -ItemType Directory -Force $dst | Out-Null
      & tar.exe -xf $zip -C $dst; if($LASTEXITCODE -ne 0){ throw "unzip failed" }
      Remove-Item $zip -Force -EA SilentlyContinue
      Write-Host "  $($m.name)/  OK" -ForegroundColor Green
    }
  } catch { $warn += "Model $($m.name): $_"; Write-Warning "  $($m.name): $_" }
}

Remove-Item $TMP -Recurse -Force -EA SilentlyContinue

Write-Host "`n== .blend files landed under models/ (verify config paths against these) ==" -ForegroundColor Cyan
Get-ChildItem "$ROOT\models" -Recurse -File -Filter *.blend | ForEach-Object { "  " + $_.FullName.Substring($ROOT.Length+1) }

if($warn.Count){ Write-Host "`n$($warn.Count) warning(s):" -ForegroundColor Yellow; $warn | ForEach-Object { Write-Host "  - $_" } }
Write-Host "`nNext:" -ForegroundColor Green
Write-Host "  pip install -r requirements.txt"
Write-Host "  python run_benchmark_suite.py --list-versions"
exit 0
