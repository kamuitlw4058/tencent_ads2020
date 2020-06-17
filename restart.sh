t=`pp tencent_ads |awk   '{printf  "%s_%s.log",\$13,\$9}'`
killp tencent_ads
mv nohup.out logs/$t
nohup python -u  $1 tencent_ads &
