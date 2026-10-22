param(
    [string]$OutputDirectory = "Example/cartpole_pino/outputs/trial_$(Get-Date -Format 'yyyyMMdd_HHmmss')",
    [string]$Python = ".venv/Scripts/python.exe"
)

$ErrorActionPreference = 'Stop'
if (Test-Path -LiteralPath $OutputDirectory) {
    throw "Use a new output directory: $OutputDirectory"
}
New-Item -ItemType Directory -Path $OutputDirectory | Out-Null
& $Python -m unittest Example.cartpole_pino.test_example -v
if ($LASTEXITCODE -ne 0) { throw 'Physics checks failed' }
& $Python -m pip freeze > "$OutputDirectory/environment.txt"
foreach ($seed in 0,1,2) {
    foreach ($kind in 'pino','fno') {
        $weight = if ($kind -eq 'pino') { 1 } else { 0 }
        & $Python -m Example.cartpole_pino.train --steps 1500 --train-samples 512 --test-samples 128 --seed $seed --physics-weight $weight --threads 2 --out "$OutputDirectory/${kind}_seed$seed" > "$OutputDirectory/${kind}_seed$seed.log"
        if ($LASTEXITCODE -ne 0) { throw "Training failed: $kind seed $seed" }
        Write-Output "$kind seed $seed complete"
    }
}
& $Python -m Example.cartpole_pino.evaluate_trial --trial $OutputDirectory > "$OutputDirectory/evaluation.log"
if ($LASTEXITCODE -ne 0) { throw 'Evaluation failed' }
& $Python -m Example.cartpole_pino.verify_resolution --trial $OutputDirectory > "$OutputDirectory/resolution.log"
if ($LASTEXITCODE -ne 0) { throw 'Resolution verification failed' }
& $Python -m Example.cartpole_pino.summarize_trial --trial $OutputDirectory
if ($LASTEXITCODE -ne 0) { throw 'Trial verification or report generation failed' }
Write-Output "Trial saved to $OutputDirectory"
