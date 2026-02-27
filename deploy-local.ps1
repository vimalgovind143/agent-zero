$ErrorActionPreference = "Stop"

$RepoPath = $PSScriptRoot
$ImageName = "agent-zero-local"
$ContainerName = "agent-zero"
$Port = 8080
$DataPath = "D:\GitHub\Agent0-Local\Data"

# Get git version info
$GitVersion = "unknown"
$GitCommitTime = "unknown"
Push-Location $RepoPath
try {
  $GitVersion = (git describe --tags --always --dirty 2>$null).Trim()
  if (-not $GitVersion) { $GitVersion = "unknown" }
  # Get commit date in short format
  $GitCommitTime = (git log -1 --format='%ad' --date=short 2>$null).Trim()
  if (-not $GitCommitTime) { $GitCommitTime = "unknown" }
} catch {
  $GitVersion = "unknown"
  $GitCommitTime = "unknown"
}
Pop-Location

Write-Host "Git Version: $GitVersion"
Write-Host "Git Commit Time: $GitCommitTime"

if ($env:A0_PORT) { $Port = [int]$env:A0_PORT }
if ($env:A0_DATA_PATH) { $DataPath = $env:A0_DATA_PATH }
if ($env:A0_CONTAINER_NAME) { $ContainerName = $env:A0_CONTAINER_NAME }
if ($env:A0_IMAGE_NAME) { $ImageName = $env:A0_IMAGE_NAME }

$existing = docker ps -a --filter "name=$ContainerName" -q
if ($existing) {
  docker stop $ContainerName | Out-Null
  docker rm $ContainerName | Out-Null
}

if (-not (Test-Path -LiteralPath $DataPath)) {
  New-Item -ItemType Directory -Path $DataPath | Out-Null
}

docker build -f DockerfileLocal -t $ImageName $RepoPath

$DataPathDocker = $DataPath -replace '\\', '/'
if ($DataPathDocker -match '^([A-Za-z]):') {
  $drive = $Matches[1].ToLower()
  $DataPathDocker = "/$drive" + ($DataPathDocker.Substring(2))
}

$ImageRef = $ImageName
if ($ImageRef -notmatch ":[^/]+$") {
  $ImageRef = "${ImageRef}:latest"
}

docker run -d --name $ContainerName -p "${Port}:80" --mount "type=bind,source=${DataPathDocker},target=/a0/usr" -e "A0_VERSION=${GitVersion}" -e "A0_COMMIT_TIME=${GitCommitTime}" $ImageRef
