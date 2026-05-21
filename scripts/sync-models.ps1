param(
    [string] $Source,
    [switch] $DownloadMissing,
    [switch] $CleanAssets
)

$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
$manifestPath = Join-Path $repoRoot "model-manifest.json"
$manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
$assetRoot = Join-Path $repoRoot ($manifest.asset_root -replace "/", [IO.Path]::DirectorySeparatorChar)
$sourceRoot = if ($Source) {
    (Resolve-Path -LiteralPath $Source).Path
} else {
    $repoRoot
}

function Join-RepoPath([string] $base, [string] $relative) {
    return Join-Path $base ($relative -replace "/", [IO.Path]::DirectorySeparatorChar)
}

function New-ParentDirectory([string] $path) {
    $parent = Split-Path -Parent $path
    if ($parent -and -not (Test-Path -LiteralPath $parent)) {
        New-Item -ItemType Directory -Path $parent | Out-Null
    }
}

function Get-ModelUrl($file) {
    if ($file.url) {
        return [string] $file.url
    }
    if ($file.repo -and $file.path) {
        $revision = if ($file.revision) { [string] $file.revision } else { "main" }
        return "https://huggingface.co/$($file.repo)/resolve/$revision/$($file.path)"
    }
    return $null
}

function Get-Sha256([string] $path) {
    return (Get-FileHash -Algorithm SHA256 -LiteralPath $path).Hash.ToLowerInvariant()
}

function Assert-ModelFileMatchesManifest($file, [string] $path) {
    $relative = [string] $file.target_path
    $expectedBytes = [int64] $file.bytes
    $actualBytes = (Get-Item -LiteralPath $path).Length
    if ($expectedBytes -gt 0 -and $actualBytes -ne $expectedBytes) {
        throw "Size mismatch for $relative`: expected $expectedBytes, got $actualBytes"
    }

    if ($file.sha256) {
        $expectedHash = ([string] $file.sha256).ToLowerInvariant()
        $actualHash = Get-Sha256 $path
        if ($actualHash -ne $expectedHash) {
            throw "SHA256 mismatch for $relative`: expected $expectedHash, got $actualHash"
        }
    }
}

function Write-HuggingFaceAccessHint([string] $url) {
    if ($url -and $url.StartsWith("https://huggingface.co/")) {
        Write-Host ""
        Write-Host "If this is a gated Hugging Face model, accept the upstream terms in the browser,"
        Write-Host "then rerun with a read token in HF_TOKEN, for example:"
        Write-Host '  $env:HF_TOKEN = "<your token>"'
        Write-Host "  .\scripts\sync-models.ps1 -DownloadMissing"
    }
}

function Try-PrepareGeneratedModel($file) {
    if ([string] $file.id -eq "asr68.bpe_vocab") {
        $vocabPath = Join-RepoPath $repoRoot ([string] $file.target_path)
        if (Test-Path -LiteralPath $vocabPath) {
            return $vocabPath
        }

        $modelRelative = "models/sherpa-onnx-zipformer-vi-2025-04-20/bpe.model"
        $modelPath = Resolve-SourceModelPath $modelRelative
        if (-not (Test-Path -LiteralPath $modelPath)) {
            return $null
        }

        New-ParentDirectory $vocabPath
        $python = Get-Command python -ErrorAction SilentlyContinue
        if (-not $python) {
            throw "Python is required to generate bpe.vocab. Install Python and run: python -m pip install sentencepiece"
        }

        $prepareScript = Join-Path $repoRoot "scripts/prepare-bpe-vocab.py"
        Write-Host "Generating $($file.id)"
        $prepareOutput = & $python.Source $prepareScript --model $modelPath --output $vocabPath 2>&1
        foreach ($line in $prepareOutput) {
            Write-Host $line
        }
        if ($LASTEXITCODE -ne 0) {
            throw "Failed to generate bpe.vocab. Install dependencies with: python -m pip install sentencepiece"
        }
        return $vocabPath
    }

    if ([string] $file.id -ne "speaker.pyannote_plda_prepared") {
        return $null
    }

    $preparedPath = Join-RepoPath $repoRoot ([string] $file.target_path)
    if (Test-Path -LiteralPath $preparedPath) {
        return $preparedPath
    }

    $pldaRelative = "models/pyannote/speaker-diarization-community-1/plda/plda.npz"
    $xvecRelative = "models/pyannote/speaker-diarization-community-1/plda/xvec_transform.npz"
    $pldaPath = Resolve-SourceModelPath $pldaRelative
    $xvecPath = Resolve-SourceModelPath $xvecRelative
    if (-not (Test-Path -LiteralPath $pldaPath) -or -not (Test-Path -LiteralPath $xvecPath)) {
        return $null
    }

    New-ParentDirectory $preparedPath
    $python = Get-Command python -ErrorAction SilentlyContinue
    if (-not $python) {
        throw "Python is required to generate Pyannote plda_prepared.npz. Install Python and run: python -m pip install numpy scipy"
    }

    $prepareScript = Join-Path $repoRoot "scripts/prepare-pyannote-plda.py"
    $pldaDir = Split-Path -Parent $pldaPath
    Write-Host "Generating $($file.id)"
    $prepareOutput = & $python.Source $prepareScript --plda-dir $pldaDir --output $preparedPath 2>&1
    foreach ($line in $prepareOutput) {
        Write-Host $line
    }
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to generate Pyannote plda_prepared.npz. Install dependencies with: python -m pip install numpy scipy"
    }
    return $preparedPath
}

function Resolve-SourceModelPath([string] $relative) {
    $candidate = Join-RepoPath $sourceRoot $relative
    $sourceLeaf = Split-Path -Leaf $sourceRoot
    if ($sourceLeaf -eq "models" -and $relative.StartsWith("models/")) {
        return Join-RepoPath $sourceRoot $relative.Substring("models/".Length)
    }
    return $candidate
}

if ($CleanAssets) {
    $assetsModels = Join-Path $assetRoot "models"
    if (Test-Path -LiteralPath $assetsModels) {
        Get-ChildItem -LiteralPath $assetsModels -Force |
            Where-Object { $_.Name -ne "README.md" } |
            Remove-Item -Recurse -Force
    }
}

$copied = 0
$downloaded = 0
$generated = 0
$missing = New-Object System.Collections.Generic.List[string]

foreach ($file in $manifest.files) {
    $relative = [string] $file.target_path
    $sourcePath = Resolve-SourceModelPath $relative
    $targetPath = Join-RepoPath $assetRoot $relative

    if (-not (Test-Path -LiteralPath $sourcePath) -and $file.generated_from) {
        $preparedPath = Try-PrepareGeneratedModel $file
        if ($preparedPath) {
            $sourcePath = $preparedPath
            $generated += 1
        } else {
            $missing.Add($relative)
            continue
        }
    }

    if (-not (Test-Path -LiteralPath $sourcePath) -and $DownloadMissing) {
        $url = Get-ModelUrl $file
        if (-not $url) {
            $missing.Add($relative)
            continue
        }

        New-ParentDirectory $sourcePath
        $tmpPath = "$sourcePath.download"
        $headers = @{}
        if ($env:HF_TOKEN -and $url.StartsWith("https://huggingface.co/")) {
            $headers["Authorization"] = "Bearer $env:HF_TOKEN"
        }

        Write-Host "Downloading $($file.id)"
        try {
            Invoke-WebRequest -Uri $url -Headers $headers -OutFile $tmpPath
            Move-Item -LiteralPath $tmpPath -Destination $sourcePath -Force
        } catch {
            if (Test-Path -LiteralPath $tmpPath) {
                Remove-Item -LiteralPath $tmpPath -Force
            }
            Write-HuggingFaceAccessHint $url
            throw
        }
        $downloaded += 1
    }

    if (-not (Test-Path -LiteralPath $sourcePath)) {
        $missing.Add($relative)
        continue
    }

    Assert-ModelFileMatchesManifest $file $sourcePath

    New-ParentDirectory $targetPath
    Copy-Item -LiteralPath $sourcePath -Destination $targetPath -Force
    $copied += 1
}

Write-Host "Copied $copied model file(s) into app assets."
if ($downloaded -gt 0) {
    Write-Host "Downloaded $downloaded missing model file(s)."
}
if ($generated -gt 0) {
    Write-Host "Generated $generated derived model file(s)."
}

if ($missing.Count -gt 0) {
    Write-Host ""
    Write-Host "Missing model file(s):"
    foreach ($item in $missing) {
        Write-Host "  - $item"
    }
    Write-Host ""
    Write-Host "Provide them under $sourceRoot or rerun with -DownloadMissing."
    exit 2
}
