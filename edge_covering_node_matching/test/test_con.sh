nohup python -u test_con.py --gpu 1 --proxy_path '../train_proxy/train_files/con/train_onlycon_lr0005/best_val_model.pth' --model_path '../train_atheta/train_files/con/train_a_onlycon/best_val_model.pth' --testset_path '../build_dataset/testset_cover/' > ./test_onglycon_cover_seed2_gpu.log  2>&1 </dev/null &