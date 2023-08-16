

upgradever=0
newver=11
for ((ver=$upgradever; $ver<$newver; ver+=1))
do
    printf -v version '%s.%1i' "$((ver/10))" "$((ver%10))"
    ./anything_tempeval.sh mscoco_var_xlan_diverse_momle_morl_lambda_2.0 $version
done