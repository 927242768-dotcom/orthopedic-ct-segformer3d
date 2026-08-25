param(
    [ValidateSet("2019", "2020", "all")]
    [string]$Edition = "2020",

    [string]$Destination = (Join-Path $PSScriptRoot "..\data\raw_public\VerSe"),

    [switch]$Download
)

$ErrorActionPreference = "Stop"

$allArchives = @(
    @{ Edition = "2019"; Name = "dataset-verse19training.zip"; Url = "https://s3.bonescreen.de/public/VerSe-complete/dataset-verse19training.zip" },
    @{ Edition = "2019"; Name = "dataset-verse19validation.zip"; Url = "https://s3.bonescreen.de/public/VerSe-complete/dataset-verse19validation.zip" },
    @{ Edition = "2019"; Name = "dataset-verse19test.zip"; Url = "https://s3.bonescreen.de/public/VerSe-complete/dataset-verse19test.zip" },
    @{ Edition = "2020"; Name = "dataset-verse20training.zip"; Url = "https://s3.bonescreen.de/public/VerSe-complete/dataset-verse20training.zip" },
    @{ Edition = "2020"; Name = "dataset-verse20validation.zip"; Url = "https://s3.bonescreen.de/public/VerSe-complete/dataset-verse20validation.zip" },
    @{ Edition = "2020"; Name = "dataset-verse20test.zip"; Url = "https://s3.bonescreen.de/public/VerSe-complete/dataset-verse20test.zip" }
)

$selected = if ($Edition -eq "all") {
    $allArchives
} else {
    $allArchives | Where-Object { $_.Edition -eq $Edition }
}

Write-Host "VerSe complete dataset download plan"
Write-Host "Edition: $Edition"
Write-Host "Destination: $Destination"
Write-Host "License reminder: maintained complete-data repository states CC BY-SA 4.0; preserve attribution/share-alike and re-check terms before redistribution."
Write-Host ""

foreach ($item in $selected) {
    Write-Host ("- {0}`n  {1}" -f $item.Name, $item.Url)
}

if (-not $Download) {
    Write-Host ""
    Write-Host "Dry plan only. Add -Download to actually download the selected large archives."
    exit 0
}

New-Item -ItemType Directory -Force -Path $Destination | Out-Null

foreach ($item in $selected) {
    $target = Join-Path $Destination $item.Name
    if (Test-Path $target) {
        Write-Host "Skip existing: $target"
        continue
    }

    Write-Host "Downloading: $($item.Name)"
    Invoke-WebRequest -Uri $item.Url -OutFile $target -UseBasicParsing
    Write-Host "Saved: $target"
}

Write-Host ""
Write-Host "Download completed. Do not commit raw archives or extracted medical images to Git."
