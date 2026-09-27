$Kernel = "mobashshirzainuddin1/amazonmlchallenge"
$OutputDir = "C:\Users\fateh\AmazonMLChallenge\output"
# matching_results.zip = leaderboard file; matcher_meta.json = validation F0.5/threshold.
# candidate_pairs.zip (~620 MB) is needed for the final submission package.
# artifacts/ (~4 GB: trained models, candidate sets, test scores) lets you reuse the run later.
$DownloadCandidates = $true
$DownloadArtifacts  = $true
$Pattern = "matching_results\.zip|matcher_meta\.json"
if ($DownloadCandidates) { $Pattern += "|candidate_pairs\.zip" }
if ($DownloadArtifacts)  { $Pattern += "|artifacts/" }
$env:PYTHONIOENCODING = "utf-8"   # the Kaggle CLI crashes printing non-ASCII log text on cp1252 consoles

New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

Write-Host "============================================" -ForegroundColor Cyan
Write-Host " Kaggle Amazon ML Challenge Watcher" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "Kernel : $Kernel"
Write-Host "Output : $OutputDir"
Write-Host ""

function Download-Output {
    # The CLI returns one page (max 200 files) per call; the notebook's /kaggle/working also
    # holds the cloned repo, so walk every page until no "Next page token" is printed.
    $token = $null
    do {
        $cliArgs = @("kernels", "output", $Kernel, "-p", $OutputDir, "--file-pattern", $Pattern, "--page-size", "200", "-o")
        if ($token) { $cliArgs += @("--page-token", $token) }
        $out = (& kaggle @cliArgs 2>&1) -join "`n"
        Write-Host $out
        $token = $null
        if ($out -match "Next page token:\s*(\S+)") { $token = $Matches[1] }
    } while ($token)
}

while ($true) {

    $time = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Write-Host "[$time] Checking Kaggle..." -ForegroundColor Cyan

    $statusText = (kaggle kernels status $Kernel 2>&1) -join "`n"
    Write-Host $statusText

    if ($statusText -match "KernelWorkerStatus\.(RUNNING|QUEUED)") {
        Write-Host "Kaggle is still running. Next check in 10 minutes." -ForegroundColor Yellow
        Write-Host ""
        Start-Sleep -Seconds 600
        continue
    }

    if ($statusText -match "KernelWorkerStatus\.COMPLETE") {
        Write-Host ""
        Write-Host " KAGGLE RUN COMPLETED - downloading" -ForegroundColor Green

        for ($attempt = 1; $attempt -le 10; $attempt++) {
            Write-Host "Download attempt $attempt/10..." -ForegroundColor Cyan
            Download-Output

            # Files keep the notebook's sub-folder: <OutputDir>\output\matching_results.zip
            $zip = Get-ChildItem -Path $OutputDir -Recurse -Filter "matching_results.zip" -ErrorAction SilentlyContinue |
                   Sort-Object LastWriteTime -Descending | Select-Object -First 1
            if ($zip -and $zip.Length -gt 0) {
                Expand-Archive -Path $zip.FullName -DestinationPath $OutputDir -Force
                $cz = Get-ChildItem -Path $OutputDir -Recurse -Filter "candidate_pairs.zip" -ErrorAction SilentlyContinue |
                      Sort-Object LastWriteTime -Descending | Select-Object -First 1
                if ($cz) { Expand-Archive -Path $cz.FullName -DestinationPath $OutputDir -Force }
                $meta = Get-ChildItem -Path $OutputDir -Recurse -Filter "matcher_meta.json" -ErrorAction SilentlyContinue |
                        Select-Object -First 1
                if ($meta) { Copy-Item $meta.FullName (Join-Path $OutputDir "matcher_meta.json") -Force }

                Write-Host ""
                Write-Host " OUTPUT DOWNLOADED: $OutputDir\matching_results.tsv" -ForegroundColor Green
                Get-ChildItem $OutputDir -File |
                    Select-Object Name, @{Name = "SizeMB"; Expression = { [math]::Round($_.Length / 1MB, 2) } }, LastWriteTime
                exit 0
            }
            Write-Host "matching_results.zip not downloaded yet. Waiting 60 seconds..." -ForegroundColor Yellow
            Start-Sleep -Seconds 60
        }
        Write-Host "Gave up after 10 attempts. Check the notebook's Output tab." -ForegroundColor Red
        exit 1
    }

    if ($statusText -match "KernelWorkerStatus\.(ERROR|FAILED|CANCEL)") {
        Write-Host " KAGGLE RUN FAILED / CANCELLED - check the notebook log (kaggle kernels logs $Kernel)" -ForegroundColor Red
        exit 1
    }

    Write-Host "Unknown status. Checking again in 2 minutes..." -ForegroundColor Yellow
    Start-Sleep -Seconds 120
}
