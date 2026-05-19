param(
    [switch] $RemoveBundledModels
)

$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
$targets = @(
    ".gradle",
    "build",
    "app/build",
    "app/.cxx"
)

foreach ($relative in $targets) {
    $path = Join-Path $repoRoot $relative
    if (Test-Path -LiteralPath $path) {
        Remove-Item -LiteralPath $path -Recurse -Force
        Write-Host "Removed $relative"
    }
}

if ($RemoveBundledModels) {
    $assetsModels = Join-Path $repoRoot "app/src/main/assets/models"
    if (Test-Path -LiteralPath $assetsModels) {
        Get-ChildItem -LiteralPath $assetsModels -Force |
            Where-Object { $_.Name -ne "README.md" } |
            Remove-Item -Recurse -Force
        Write-Host "Removed bundled model assets"
    }
}
