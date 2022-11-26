#!/bin/bash 
URL="https://10.108.255.249/include/auth_action.php" 
username=21210240166
password=fDu180038
ip=10.176.50.17
curl $URL --insecure --data "action=login&username=$username&password=$password&ac_id=1&user_ip=$ip&nas_ip=&user_mac=&save_me=1&ajax=1" > /dev/null 2>&1
