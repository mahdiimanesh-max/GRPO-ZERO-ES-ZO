#!/usr/bin/expect -f
# Setup script using expect for password handling

set timeout 60
set remote_user "d3"
set remote_host "100.89.206.96"
set remote_path "/home/d3/es-fine-tuning-paper"
if {![info exists env(SSH_PASSWORD)]} {
    puts stderr "Set SSH_PASSWORD in the environment (never commit it)."
    exit 1
}
set password $env(SSH_PASSWORD)

# Step 1: Copy files using rsync
spawn rsync -avz --progress --exclude 'es/' --exclude '__pycache__/' --exclude '*.pyc' --exclude '.git/' --exclude 'checkpoint_*' --exclude 'finetuned_*' --exclude 'results_*' --exclude 'logs_*' --exclude 'ckpt_*' /Users/mehdiiranmanesh/Desktop/es-fine-tuning-paper/ $remote_user@$remote_host:$remote_path/

expect {
    "password:" {
        send "$password\r"
        exp_continue
    }
    "Permission denied" {
        puts "ERROR: Authentication failed"
        exit 1
    }
    eof
}

# Step 2: SSH and set up environment
spawn ssh $remote_user@$remote_host

expect {
    "password:" {
        send "$password\r"
    }
    "Permission denied" {
        puts "ERROR: Authentication failed"
        exit 1
    }
}

expect "$ "
send "cd $remote_path\r"
expect "$ "

send "python3 -m venv es\r"
expect "$ "

send "source es/bin/activate\r"
expect "(es)"

send "pip install --upgrade pip\r"
expect "(es)"

send "pip install -r requirement.txt\r"
expect "(es)"

send "pip install tensorboard pyarrow jinja2 tokenizers safetensors pyyaml\r"
expect "(es)"

send "chmod +x run_es_gpu.sh run_zo_gpu.sh run_grpo_gpu.sh\r"
expect "(es)"

send "python -c \"import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')\"\r"
expect "(es)"

send "exit\r"
expect eof

puts "\nSetup complete!"
