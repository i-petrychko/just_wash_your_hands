#!/bin/bash
sudo mkdir /actions-runner;
cd /actions-runner;

echo Downloading GitHub Actions Runner...;
sudo curl -o actions-runner-linux-x64-2.317.0.tar.gz -L https://github.com/actions/runner/releases/download/v2.317.0/actions-runner-linux-x64-2.317.0.tar.gz;
sudo tar xzf ./actions-runner-linux-x64-2.317.0.tar.gz;

echo Registering the runner with token $TOKEN;
export RUNNER_ALLOW_RUNASROOT=1;
sudo yum install -y dotnet-sdk-6.0
sudo -E ./config.sh --url https://github.com/i-petrychko/just_wash_your_hands --token $TOKEN --unattended --ephemeral --labels training-runner;

echo Starting the runner...;
sudo -E ./run.sh;