# Train the Yelp2018 V0-V3 factorial (RQ5) with temperature_user=0.15, the main
# configuration's value (main paper Table 6), so that V3 has the configuration
# of the TaxPro-CL-main result. Idempotent: completed seeds are skipped.
# Outputs: log/p0/taxprocl/yelp2018/A2-V{0-3}-tempuser0.15/seed{42,0,1}.

$ErrorActionPreference = "Stop"
$logFile = "results/yelp_factorial_progress.log"
function Log($msg) {
    $line = "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') | $msg"
    Write-Output $line
    Add-Content -Path $logFile -Value $line
}

$variants = @(
  @{ id="A2-V0-tempuser0.15"; direction="random";   adaptive="false" },
  @{ id="A2-V1-tempuser0.15"; direction="taxonomy"; adaptive="false" },
  @{ id="A2-V2-tempuser0.15"; direction="random";   adaptive="true"  },
  @{ id="A2-V3-tempuser0.15"; direction="taxonomy"; adaptive="true"  }
)

Log "START yelp2018 factorial, temperature_user=0.15, 4 variants x 3 seeds = 12 runs"

foreach ($v in $variants) {
  foreach ($seed in 42,0,1) {
    $outDir = "log/p0/taxprocl/yelp2018/$($v.id)/seed$seed"
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
    python main.py --model TaxPro-CL --dataset yelp2018 --seed $seed `
      --augmentation_direction $v.direction --use_adaptive_epsilon $v.adaptive `
      --temperature 0.125 --temperature_user 0.15 --taxonomy_policy no_merge --device cuda
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

Log "ALL DONE: yelp2018 factorial, temperature_user=0.15"
