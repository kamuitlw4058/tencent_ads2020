t=`cat last_cmd`
killp $t
echo $1 > last_cmd
d=`date "+%Y%m%d%H%M%S"`
nohup python -u  $1 tencent_ads >logs/$1_$d.log & 
