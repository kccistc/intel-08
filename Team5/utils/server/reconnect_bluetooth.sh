echo "Restarting PipeWire services for clean audio output..."
systemctl --user restart pipewire pipewire-pulse wireplumber
