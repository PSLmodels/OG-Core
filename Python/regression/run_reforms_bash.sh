#!/bin/bash
# TODO also these reforms: ${r}1 ${r}2 ${r}3 ${r}4 ${r}5 ${r}6 ${r}7 ${r}8 ${r}9 t1 t2

export JENKINS_DOMAIN="http://54.159.16.16:8080"
export OGUSA_BRANCH=$(git branch | grep "*" | sed 's/ //' | sed 's/\*//')
export r=reform
submit_jobs(){
    for reform in ${r}0 ;
       do
          export token_part="token=${REMOTE_BUILD_TOKEN}"
          export cause="cause=Cause+CI+Build"
          export branch="ogusabranch=${OGUSA_BRANCH}"
          export middle="/job/ci-mode-${reform}/build"
          export JENKINS_URL="${JENKINS_DOMAIN}${middle}?${token_part}&${branch}&{cause}"
          echo Attempt to curl $JENKINS_URL with OSPC_API_KEY secret
          if [ "$OSPC_API_KEY" = "" ];then
              echo OSPC_API_KEY is empty - wont work
    	  fi
	   curl --user ospctaxbrain:$OSPC_API_KEY "$JENKINS_URL" || return 1;
       done
}
submit_jobs || echo "Failed on $JENKINS_URL"

