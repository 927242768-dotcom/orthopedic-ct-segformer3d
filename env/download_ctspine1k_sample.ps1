param(
    [string]$RepoId = "alexanderdann/CTSpine1K",

    [ValidateSet("MSD-T10", "COVID-19", "COLONOG", "HNSCC-3DCT-RT")]
    [string]$SubDataset = "MSD-T10",

    [string[]]$CaseIds = @("liver_169", "liver_0", "liver_1"),

    [string]$Destination = (Join-Path $PSScriptRoot "..\data\raw_public\CTSpine1K"),

    [switch]$Download
)

$ErrorActionPreference = "Stop"

function Test-GzipFile {
    param([Parameter(Mandatory = $true)][string]$Path)

    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        return $false
    }

    $fileStream = $null
    $gzipStream = $null
    try {
        $fileStream = [System.IO.File]::OpenRead($Path)
        $gzipStream = [System.IO.Compression.GZipStream]::new(
            $fileStream,
            [System.IO.Compression.CompressionMode]::Decompress
        )
        $buffer = New-Object byte[] (1024 * 1024)
        while ($gzipStream.Read($buffer, 0, $buffer.Length) -gt 0) {
            # Read to EOF so truncated gzip streams are detected before a file is trusted.
        }
        return $true
    } catch {
        Write-Warning "Gzip integrity check failed: $Path ($($_.Exception.Message))"
        return $false
    } finally {
        if ($null -ne $gzipStream) {
            $gzipStream.Dispose()
        }
        if ($null -ne $fileStream) {
            $fileStream.Dispose()
        }
    }
}

function Save-GzipDownloadAtomic {
    param(
        [Parameter(Mandatory = $true)][string]$Uri,
        [Parameter(Mandatory = $true)][string]$Target,
        [Parameter(Mandatory = $true)][string]$Description
    )

    if (Test-Path -LiteralPath $Target -PathType Leaf) {
        if (Test-GzipFile -Path $Target) {
            Write-Host "Skip existing verified $Description`: $Target"
            return
        }
        Write-Warning "Replacing incomplete $Description`: $Target"
        Remove-Item -LiteralPath $Target -Force
    }

    $part = "$Target.part"
    if (Test-Path -LiteralPath $part) {
        Write-Warning "Removing stale partial download: $part"
        Remove-Item -LiteralPath $part -Force
    }

    try {
        Write-Host "Downloading $Description"
        Invoke-WebRequest -Uri $Uri -OutFile $part -UseBasicParsing
        if (-not (Test-GzipFile -Path $part)) {
            throw "Downloaded gzip failed integrity validation: $Uri"
        }
        Move-Item -LiteralPath $part -Destination $Target -Force
    } finally {
        if (Test-Path -LiteralPath $part) {
            Remove-Item -LiteralPath $part -Force
        }
    }
}

if ($CaseIds.Count -eq 0) {
    throw "CaseIds 不能为空"
}

$baseUrl = "https://huggingface.co/datasets/$RepoId/resolve/main/raw_data"
$splitUrl = "https://raw.githubusercontent.com/MIRACLE-Center/CTSpine1K/main/data_split.txt"
$subRoot = Join-Path $Destination $SubDataset
$volumeDir = Join-Path $subRoot "volumes"
$labelDir = Join-Path $subRoot "labels"

$plan = @()
foreach ($caseId in $CaseIds) {
    if ([string]::IsNullOrWhiteSpace($caseId)) {
        throw "CaseIds 中包含空值"
    }
    $volumeName = "$caseId.nii.gz"
    $labelName = "${caseId}_seg.nii.gz"
    $plan += [pscustomobject]@{
        CaseId = $caseId
        VolumeName = $volumeName
        VolumeUrl = "$baseUrl/volumes/$SubDataset/$volumeName"
        LabelName = $labelName
        LabelUrl = "$baseUrl/labels/$SubDataset/$labelName"
    }
}

Write-Host "CTSpine1K small-sample download plan"
Write-Host "Repository: $RepoId"
Write-Host "Sub-dataset: $SubDataset"
Write-Host "Destination: $subRoot"
Write-Host "Purpose: real-data preprocessing/QC smoke test only; do not report model metrics from this convenience subset."
Write-Host "License reminder: preserve the original CTSpine1K/sub-dataset terms and attribution; verify terms again before redistribution."
Write-Host ""

foreach ($item in $plan) {
    Write-Host ("- {0}`n  CT:    {1}`n  label: {2}" -f $item.CaseId, $item.VolumeUrl, $item.LabelUrl)
}
Write-Host "- official split metadata"
Write-Host "  $splitUrl"

if (-not $Download) {
    Write-Host ""
    Write-Host "Dry plan only. Add -Download to download the selected NIfTI pairs."
    exit 0
}

New-Item -ItemType Directory -Force -Path $volumeDir | Out-Null
New-Item -ItemType Directory -Force -Path $labelDir | Out-Null

$downloaded = @()
foreach ($item in $plan) {
    $volumeTarget = Join-Path $volumeDir $item.VolumeName
    $labelTarget = Join-Path $labelDir $item.LabelName

    Save-GzipDownloadAtomic -Uri $item.VolumeUrl -Target $volumeTarget -Description "CT $($item.CaseId)"
    Save-GzipDownloadAtomic -Uri $item.LabelUrl -Target $labelTarget -Description "label $($item.CaseId)"

    $downloaded += [pscustomobject]@{
        case_id = $item.CaseId
        volume = $volumeTarget
        label = $labelTarget
        volume_url = $item.VolumeUrl
        label_url = $item.LabelUrl
        volume_size_bytes = (Get-Item -LiteralPath $volumeTarget).Length
        label_size_bytes = (Get-Item -LiteralPath $labelTarget).Length
        gzip_integrity_checked = $true
    }
}

$splitTarget = Join-Path $Destination "data_split.txt"
if (-not (Test-Path $splitTarget)) {
    Write-Host "Downloading official split metadata"
    Invoke-WebRequest -Uri $splitUrl -OutFile $splitTarget -UseBasicParsing
} else {
    Write-Host "Skip existing split metadata: $splitTarget"
}

$manifest = [ordered]@{
    dataset = "CTSpine1K"
    repository = $RepoId
    sub_dataset = $SubDataset
    downloaded_at = (Get-Date).ToString("o")
    intended_use = "real-data preprocessing/QC smoke test"
    official_split_file = $splitTarget
    cases = $downloaded
}
$manifestPath = Join-Path $subRoot "download_manifest.json"
$manifest | ConvertTo-Json -Depth 6 | Set-Content -Path $manifestPath -Encoding UTF8

Write-Host ""
Write-Host "Download completed: $subRoot"
Write-Host "Manifest: $manifestPath"
Write-Host "Do not commit raw medical images, labels, or downloaded archives to Git."
