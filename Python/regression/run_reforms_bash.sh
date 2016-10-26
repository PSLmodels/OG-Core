#!/bin/bash
# TODO also these reforms: ${r}1 ${r}2 ${r}3 ${r}4 ${r}5 ${r}6 ${r}7 ${r}8 ${r}9 t1 t2

export JENKINS_DOMAIN="http://54.159.16.16:8080"

export r=reform
submit_jobs(){
    for reform in ${r}0 ;
       do
          export token_part="?token=${REMOTE_BUILD_TOKEN}&cause=Cause+CI+Build"
          export JENKINS_URL="${JENKINS_DOMAIN}/job/ci-mode-${reform}/build${token_part}"
          echo Attempt to curl $JENKINS_URL with OSPC_API_KEY secret
          if [ "$OSPC_API_KEY" = "" ];then
              echo OSPC_API_KEY is empty - wont work
    	  fi
	   curl --user ospctaxbrain:$OSPC_API_KEY "$JENKINS_URL" || return 1;
       done
}
submit_jobs || echo "Failed on $JENKINS_URL"

