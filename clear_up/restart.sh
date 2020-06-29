t=`cat last_cmd`
killp $t
d=`date "+%Y%m%d%H%M%S"`
mv nohup.out logs/$t_$d.log

echo $1 > last_cmd
nohup python -u  $1 tencent_ads  & 
