
# $1 gpuid
# $2 runid

# base方法
svroot=saved-digit/base_run${2}
python3 main_base.py --gpu $1 --data mnist --epochs 50 --nbatch 100 --lr 1e-4 --svroot $svroot
python3 main_test_digit.py --gpu $1 --modelpath $svroot/best.pkl --svpath $svroot/test.log


