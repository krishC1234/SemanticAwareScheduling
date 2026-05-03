#!/bin/bash
set -e

echo "==> Installing system packages and SLURM..."
apt-get update
apt-get install -y munge slurm-wlm slurm-wlm-basic-plugins

echo "==> Installing Python dependencies..."
pip install -r requirements.txt

# Create required directories
mkdir -p /var/spool/slurmd /var/spool/slurmctld /var/log/slurm /var/run/munge /etc/slurm

# Fix ownership (must be after package install so users exist)
chown slurm:slurm /var/spool/slurmd /var/spool/slurmctld /var/log/slurm
chown munge:munge /var/run/munge /var/log/munge /etc/munge

# Generate munge key if missing
if [ ! -f /etc/munge/munge.key ]; then
    echo "==> Creating munge key..."
    dd if=/dev/urandom bs=1 count=1024 > /etc/munge/munge.key 2>/dev/null
    chown munge:munge /etc/munge/munge.key
    chmod 400 /etc/munge/munge.key
fi

# Detect hardware
HOST=$(hostname)
CPUS=$(nproc)
GPU_COUNT=$(nvidia-smi -L 2>/dev/null | wc -l || echo 0)
RAM_MB=$(free -m | awk '/Mem:/ {print $2}')

echo "==> Detected: $CPUS CPUs, $GPU_COUNT GPUs, ${RAM_MB}MB RAM, hostname=$HOST"

# Write slurm.conf
cat > /etc/slurm/slurm.conf <<SLURM
ClusterName=local
SlurmctldHost=$HOST

AuthType=auth/munge
SchedulerType=sched/backfill
SelectType=select/cons_tres
SelectTypeParameters=CR_Core_Memory

ProctrackType=proctrack/linuxproc
ReturnToService=2
SlurmctldTimeout=300
SlurmdTimeout=300

SlurmctldPort=6817
SlurmdPort=6818

SlurmctldLogFile=/var/log/slurm/slurmctld.log
SlurmdLogFile=/var/log/slurm/slurmd.log
SlurmdSpoolDir=/var/spool/slurmd
StateSaveLocation=/var/spool/slurmctld

GresTypes=gpu

NodeName=$HOST CPUs=$CPUS RealMemory=$RAM_MB Gres=gpu:$GPU_COUNT State=UNKNOWN
PartitionName=gpu Nodes=$HOST Default=YES MaxTime=INFINITE State=UP
SLURM

# Write gres.conf for GPUs
if [ "$GPU_COUNT" -gt 0 ]; then
    GPU_LAST=$((GPU_COUNT - 1))
    cat > /etc/slurm/gres.conf <<GRES
NodeName=$HOST Name=gpu Type=nvidia File=/dev/nvidia[0-$GPU_LAST]
GRES
else
    echo "# No GPUs detected" > /etc/slurm/gres.conf
fi

# Start munge
echo "==> Starting munge..."
pkill munged 2>/dev/null || true
sleep 1
munged --force
sleep 1

echo "==> Testing munge..."
munge -n | unmunge | head -3

# Start SLURM
echo "==> Starting SLURM..."
pkill slurmctld 2>/dev/null || true
pkill slurmd 2>/dev/null || true
sleep 1

slurmctld
sleep 3
slurmd
sleep 2

# Set node to ready
scontrol update NodeName=$HOST State=UNDRAIN 2>/dev/null || true

# Show status
echo ""
echo "==> SLURM ready:"
sinfo
echo ""

echo "==> Test with:"
echo "    sbatch --gres=gpu:1 --wrap='nvidia-smi && sleep 5'"
echo "    squeue"
