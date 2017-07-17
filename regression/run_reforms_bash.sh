#!/bin/bash

submit_jobs(){
    export REFORMS_TO_RUN=$(cat ../.regression.txt | grep reforms_to_run | sed 's/reforms_to_run//');
    if [ "${REFORMS_TO_RUN}" = "" ];then
        echo Error - Must set REFORMS_TO_RUN env var - space separated list of reform strings;
    else

        for reform in $(echo ${REFORMS_TO_RUN} | tr " " "\n");
           do
              export BUILD_CAUSE="$VERSION"
              export token_part="token=${REMOTE_BUILD_TOKEN}";
              export cause="cause=$(echo $BUILD_CAUSE | tr " " "+")";
              export branch="ogusabranch=${BRANCH_NAME}";
              export middle="/job/ci-mode-simple-${reform}/buildWithParameters";
              export JENKINS_URL="${JENKINS_DOMAIN}${middle}?${token_part}&${branch}&${cause}";
              echo Attempt to curl $JENKINS_URL with OSPC_API_KEY secret
              if [ "$OSPC_API_KEY" = "" ];then
                  echo OSPC_API_KEY is empty - wont work;
        	  fi
    	   curl --user ospctaxbrain:$OSPC_API_KEY "$JENKINS_URL" || return 1;
           done
    fi
}
setup_miniconda(){
    wget http://repo.continuum.io/miniconda/Miniconda-latest-Linux-x86_64.sh -O miniconda.sh;
    rm -rf miniconda;
    rm -rf OG-USA;
    bash miniconda.sh -b -p miniconda;
    export PATH=`pwd`/miniconda/bin:$PATH;
    conda config --set always_yes yes --set changeps1 no;
    conda install beautifulsoup4 lxml requests pandas anaconda-client;
}
poll(){
    echo Begin Polling && python polling_jobs.py ${REFORMS_TO_RUN} && echo End Polling;
}
push_artifacts(){
    echo Push artifacts;
    export org=opensourcepolicycenter;
    export pkg=ogusaregression;
    export summary="Regression artifacts from $VERSION";
    mv artifacts $VERSION
    echo tar cjvf ${VERSION}.tar.bz2 ${VERSION}/* \&\& anaconda --token ${ANACONDA_OSPC_TOKEN} upload --user $org --version $VERSION --package $pkg --package-type file --summary \"$summary\" ${VERSION}.tar.bz2;
    tar cjvf ${VERSION}.tar.bz2 ${VERSION}/* && anaconda --token ${ANACONDA_OSPC_TOKEN} upload --user $org --version $VERSION --package $pkg --package-type file --summary "$summary" ${VERSION}.tar.bz2;
}
export VERSION="${BRANCH_NAME} $(date)";
export VERSION=$(echo $VERSION | sed 's/:/_/g' | sed 's/ /_/g');

echo Submit REFORMS_TO_RUN: $REFORMS_TO_RUN
set +x
rm -rf artifacts && submit_jobs && setup_miniconda && poll && push_artifacts
set -x
