# Run V0/V1/V2 of the Amazon-Book direction-by-magnitude factorial (RQ5) under
# taxonomy_policy=merge_t10 (the main configuration's policy). V3 is not
# trained by this script: A2-V3-warm20, an independent run of the main
# configuration made separately on a second machine (ESM Table S23), serves as
# V3; its test metrics are within 0.26% of the TaxPro-CL-main checkpoint's.
# Output goes to NEW directories (A2-V{0,1,2}-mergedt10), so earlier runs are
# never overwritten.

$ErrorActionPreference = "Stop"
$logFile = "results/amazonbook_factorial_mergedt10_progress.log"
function Log($msg) {
    $line = "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') | $msg"
    Write-Output $line
    Add-Content -Path $logFile -Value $line
}

$variants = @(
  @{ id="A2-V0-mergedt10"; direction="random";   adaptive="false" },
  @{ id="A2-V1-mergedt10"; direction="taxonomy"; adaptive="false" },
  @{ id="A2-V2-mergedt10"; direction="random";   adaptive="true"  }
)

Log "START amazon-book factorial re-run, taxonomy_policy=merge_t10, 3 variants x 3 seeds = 9 runs (V3 reused from TaxPro-CL-main, not re-run)"

foreach ($v in $variants) {
  foreach ($seed in 42,0,1) {
    $outDir = "log/p0/taxprocl/amazon-book/$($v.id)/seed$seed"
    if (Test-Path "$outDir/run_manifest.json") {
      $status = (Get-Content "$outDir/run_manifest.json" | ConvertFrom-Json).status
      if ($status -eq "completed") {
        Log "SKIP (already completed): $($v.id) seed=$seed"
        continue
      }
    }
    Log "RUN: $($v.id) seed=$seed direction=$($v.direction) adaptive=$($v.adaptive)"
    $env:TAXPROCL_RUN_OUTPUT_DIR = $outDir
    $start = Get-Date
    python main.py --model TaxPro-CL --dataset amazon-book --seed $seed `
      --augmentation_direction $v.direction --use_adaptive_epsilon $v.adaptive `
      --temperature 0.1 --temperature_user 0.2 --epsilon_max 0.2 `
      --gamma_cold 1.5 --gamma_warm 1.0 --warm_start_epochs 20 --mu 0.9 `
      --taxonomy_policy merge_t10 --device cuda
    $exitCode = $LASTEXITCODE
    $elapsed = [int]((Get-Date) - $start).TotalSeconds
    Remove-Item Env:\TAXPROCL_RUN_OUTPUT_DIR -ErrorAction SilentlyContinue
    if ($exitCode -ne 0) {
      Log "FAILED ($($elapsed)s, exit=$exitCode): $($v.id) seed=$seed"
      throw "main.py failed for $($v.id) seed=$seed"
    }
    Log "DONE ($($elapsed)s): $($v.id) seed=$seed"
  }
}

Log "ALL DONE: amazon-book factorial re-run with taxonomy_policy=merge_t10"
