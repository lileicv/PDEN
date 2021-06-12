
# ifconfig 查询本机 ip
local_ip=`ifconfig -a|grep inet|grep -v inet6`

for remote_ip in 10.25.0.24 10.206.32.10 
do
    echo "####################################################"
    echo "本机代码同步至 ==> $remote_ip"
    if [[ $local_ip =~ $remote_ip ]]; then
        echo "本机ip，取消同步"
    else
        sshpass -p zlyai1991 rsync -avz --exclude bak --exclude saved* --exclude log --exclude __pycache__ ./ lilei@$remote_ip:/home/lilei/dg-pden-digit
    fi
done

