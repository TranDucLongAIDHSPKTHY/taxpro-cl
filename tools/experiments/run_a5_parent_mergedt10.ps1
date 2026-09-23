# A5 (leaf vs. parent prototype granularity), Amazon-Book: parent-granularity
# run under taxonomy_policy=merge_t10, the main configuration's policy, so that
# the comparator (TaxPro-CL-main, prototype_mode=leaf) differs from these runs
# in prototype_mode only. Every other hyperparameter equals the main
# configuration (configure/TaxPro-CL.txt defaults, spelled out below).
# Output goes to log/p0/taxprocl/amazon-book/ablation-A5-parent-mergedt10/.

$ErrorActionPreference = "Stop"
$logFile = "results/a5_parent_mergedt10_progress.log"
function Log($msg) {
    $line = "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') | $msg"
    Write-Output $line
    Add-Content -Path $logFile -Value $line
}

Log "START amazon-book A5 parent-granularity runs, taxonomy_policy=merge_t10, 3 seeds"

foreach ($seed in 42,0,1) {
    $outDir = "log/p0/taxprocl/amazon-book/ablation-A5-parent-mergedt10/seed$seed"
    if (Test-Path "$outDir/run_manifest.json") {
        $status = (Get-Content "$outDir/run_manifest.json" | ConvertFrom-Json).status
        if ($status -eq "completed") {
            Log "SKIP (already completed): seed=$seed"
            continue
        }
    }
    Log "RUN: A5-parent-mergedt10 seed=$seed"
    $env:TAXPROCL_RUN_OUTPUT_DIR = $outDir
    $start = Get-Date
    python main.py --model TaxPro-CL --dataset amazon-book --seed $seed `
      --prototype_mode parent --taxonomy_policy merge_t10 `
      --temperature 0.1 --temperature_user 0.2 --epsilon_max 0.2 `
      --gamma_cold 1.5 --gamma_warm 1.0 --warm_start_epochs 20 --mu 0.9 `
      --device cuda
    $exitCode = $LASTEXITCODE
    $elapsed = [int]((Get-Date) - $start).TotalSeconds
    Remove-Item Env:\TAXPROCL_RUN_OUTPUT_DIR -ErrorAction SilentlyContinue
    if ($exitCode -ne 0) {
        Log "FAILED ($($elapsed)s, exit=$exitCode): seed=$seed"
        throw "main.py failed for seed=$seed"
    }
    Log "DONE ($($elapsed)s): seed=$seed"
}

Log "ALL DONE: amazon-book A5 parent-granularity runs under merge_t10"
