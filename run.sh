# 1

python fit_data.py --type 'vox'
python fit_data.py --type 'point'
python fit_data.py --type 'mesh'

## Training

# 2.1

python train_model.py --type 'vox' --lr 4e-4 --save_freq 100 --batch_size 32 --max_iter 10001

# 2.2

python train_model.py --type 'point' --lr 4e-4 --save_freq 100 --batch_size 32 --max_iter 10001 --n_points 5000

# 2.3

python train_model.py --type 'vox' --lr 4e-4 --save_freq 100 --batch_size 32 --max_iter 10001

# 2.5

wSmooth=('0' '0.05' '0.1' '0.5' '1')

for i in "${wSmooth[@]}"; do
    python train_model.py --type 'mesh' --lr 4e-4 --save_freq 100 --batch_size 32 --max_iter 2001 --w_smooth $i --checkpoint_suffix _w$i
done

# 3.3

python train_model.py --type 'mesh' --lr 4e-4 --save_freq 100 --batch_size 32 --max_iter 1001 --checkpoint_suffix _full
python train_model.py --type 'point' --lr 4e-4 --save_freq 100 --batch_size 32 --max_iter 1001 --checkpoint_suffix _full --n_points 5000
python train_model.py --type 'vox' --lr 4e-4 --save_freq 100 --batch_size 32 --max_iter 1001 --checkpoint_suffix _full

## Evaluation

# 2.1

python eval_model.py --type 'vox'  --load_checkpoint --vis_freq 100

# 2.2

python eval_model.py --type 'point'  --load_checkpoint --vis_freq 100

# 2.3

python eval_model.py --type 'mesh'  --load_checkpoint --vis_freq 100

# 2.5

w=("0" "0.05" "0.1" "0.5" "1")

for i in "${w[@]}"; do
    python eval_model.py --type 'mesh'  --load_checkpoint --vis_freq 100 --checkpoint_suffix _w$i
done

# 3.3

type=("mesh" "point" "vox")

for i in "${type[@]}"; do
    python eval_model.py --type  $i --load_checkpoint --vis_freq 100 --checkpoint_suffix _full
done
