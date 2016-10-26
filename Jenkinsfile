node {
    withCredentials([[$class: 'StringBinding', credentialsId: 'OSPC_API_KEY', variable: 'OSPC_API_KEY']]) {
        withCredentials([[$class: 'StringBinding', credentialsId: 'REMOTE_BUILD_TOKEN', variable: 'REMOTE_BUILD_TOKEN']]) {
            sh '''if [ "$REMOTE_BUILD_TOKEN" = "" ];then echo WONT WORK - REMOTE_BUILD_TOKEN not there;fi '''
            sh '''if [ "$OSPC_API_KEY" = "" ];then echo WONT WORK - OSPC_API_KEY not there;fi '''
            sh '''echo About to checkout scm'''
            checkout scm
            sh '''echo Checked out scm'''

            sh '''export OGUSA_BRANCH=$(git branch | grep "*" | sed 's/ //' | sed 's/\*//') && cd Python/regression && bash run_reforms_bash.sh'''
        }
    }
}
