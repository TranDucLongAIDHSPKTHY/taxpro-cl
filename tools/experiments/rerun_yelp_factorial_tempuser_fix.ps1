# Re-run the V0-V3 factorial on Yelp2018 with temperature_user=0.15 (matching the
# main-configuration value in Table 6), fixing a confound found during a
# cross-check: the original A2-V0..V3 Yelp2018 runs used temperature_user's
# base-config default (0.2) instead of Yelp2018's tuned value (0.15), so "V3" did
# not exactly match the TaxPro-CL-main checkpoint used everywhere else in the paper.
# Output goes to a NEW directory (A2-V{0-3}-tempuser0.15) so the original
# (documented-as-flawed) A2-V0..V3 runs are preserved for the audit trail.

$ErrorActionPreference = "Stop"
$logFile = "results/yelp_factorial_tempuser_fix_progress.log"
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

Log "START yelp2018 factorial re-run, temperature_user=0.15 fix, 4 variants x 3 seeds = 12 runs"

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

Log "ALL DONE: yelp2018 factorial re-run with temperature_user=0.15"
