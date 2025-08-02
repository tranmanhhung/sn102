# PM2 Commands for BetterTherapy Miners

## Cài đặt PM2 (nếu chưa có)
```bash
npm install -g pm2
```

## Khởi chạy miner với PM2

### 1. Chạy OptimizedMiner (RECOMMENDED)
```bash
# Chạy với cấu hình từ ecosystem.config.js
pm2 start ecosystem.config.js --only optimized-miner

# Hoặc chạy trực tiếp
pm2 start run_optimized_miner.py --name "optimized-miner" --interpreter python3 -- --config mid_range --model balanced

# Chạy với cấu hình khác
pm2 start run_optimized_miner.py --name "speed-miner" --interpreter python3 -- --config low_end --model speed
pm2 start run_optimized_miner.py --name "quality-miner" --interpreter python3 -- --config high_end --model quality
```

### 2. Chạy Standard Miner
```bash
# Chạy miner gốc
pm2 start ecosystem.config.js --only standard-miner

# Hoặc trực tiếp
pm2 start neurons/miner.py --name "standard-miner" --interpreter python3 -- --netuid 34 --subtensor.network finney --logging.debug
```

## Quản lý Processes

### Xem trạng thái
```bash
# Xem tất cả processes
pm2 list

# Xem thông tin chi tiết
pm2 show optimized-miner

# Monitor real-time
pm2 monit
```

### Dừng/Khởi động lại
```bash
# Dừng miner
pm2 stop optimized-miner

# Khởi động lại
pm2 restart optimized-miner

# Reload (zero-downtime restart)
pm2 reload optimized-miner

# Xóa process
pm2 delete optimized-miner
```

### Quản lý tất cả processes
```bash
# Dừng tất cả
pm2 stop all

# Khởi động lại tất cả
pm2 restart all

# Xóa tất cả
pm2 delete all
```

## Logs

### Xem logs
```bash
# Xem logs real-time
pm2 logs optimized-miner

# Xem logs của tất cả processes
pm2 logs

# Xem logs với số dòng giới hạn
pm2 logs optimized-miner --lines 100

# Xóa logs
pm2 flush
```

### Vị trí log files
- Error logs: `./logs/optimized-miner-error.log`
- Output logs: `./logs/optimized-miner-out.log`
- Combined logs: `./logs/optimized-miner-combined.log`

## Startup Scripts

### Tự động khởi động khi boot
```bash
# Lưu cấu hình hiện tại
pm2 save

# Tạo startup script
pm2 startup

# Chạy lệnh được gợi ý (thường cần sudo)
# Ví dụ: sudo env PATH=$PATH:/usr/bin /usr/lib/node_modules/pm2/bin/pm2 startup systemd -u hungtranmanh --hp /Users/hungtranmanh
```

### Khôi phục processes sau reboot
```bash
# Khôi phục tất cả processes đã save
pm2 resurrect
```

## Cấu hình nâng cao

### Chạy nhiều instances (cluster mode)
```bash
# Chạy 2 instances của optimized miner
pm2 start run_optimized_miner.py --name "optimized-miner-cluster" --instances 2 --interpreter python3
```

### Giới hạn memory và tự động restart
```bash
pm2 start run_optimized_miner.py --name "optimized-miner" --max-memory-restart 4G --interpreter python3
```

### Watch file changes (development)
```bash
pm2 start run_optimized_miner.py --name "dev-miner" --watch --interpreter python3
```

## Troubleshooting

### Miner không start
```bash
# Xem chi tiết lỗi
pm2 show optimized-miner

# Xem logs
pm2 logs optimized-miner --err

# Restart với force
pm2 restart optimized-miner --force
```

### High memory usage
```bash
# Monitor resource usage
pm2 monit

# Restart nếu cần
pm2 restart optimized-miner
```

### Logs quá lớn
```bash
# Cài đặt log rotation
pm2 install pm2-logrotate

# Cấu hình rotation
pm2 set pm2-logrotate:max_size 10M
pm2 set pm2-logrotate:retain 30
pm2 set pm2-logrotate:compress true
```

## Quick Reference

| Command | Description |
|---------|-------------|
| `pm2 start ecosystem.config.js --only optimized-miner` | Khởi chạy OptimizedMiner |
| `pm2 list` | Xem tất cả processes |
| `pm2 logs optimized-miner` | Xem logs real-time |
| `pm2 monit` | Monitor dashboard |
| `pm2 restart optimized-miner` | Restart miner |
| `pm2 stop optimized-miner` | Dừng miner |
| `pm2 save` | Lưu cấu hình hiện tại |

## Recommended Usage

1. **Production**: `pm2 start ecosystem.config.js --only optimized-miner`
2. **Development**: `pm2 start run_optimized_miner.py --name dev-miner --watch --interpreter python3`
3. **Testing**: Chạy trực tiếp `python run_optimized_miner.py` để debug

Với PM2, miner sẽ tự động restart nếu crash và có thể monitor performance dễ dàng!
