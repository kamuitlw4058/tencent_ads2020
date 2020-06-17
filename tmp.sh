#killp tencent_ads
t=`pp tencent_ads |awk   '{printf  "%s_%s.log",$13,$9}'`
#mv nohup.out logs/$t
#nohup python -u  $1 tencent_ads &
