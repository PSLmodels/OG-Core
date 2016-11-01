
node {
    withCredentials([[$class: 'StringBinding', credentialsId: 'OSPC_API_KEY', variable: 'OSPC_API_KEY']]) {
        withCredentials([[$class: 'StringBinding', credentialsId: 'REMOTE_BUILD_TOKEN', variable: 'REMOTE_BUILD_TOKEN']]) {
            withCredentials([[$class: 'StringBinding', credentialsId: 'ANACONDA_OSPC_TOKEN', variable: 'ANACONDA_OSPC_TOKEN']]) {
                 withCredentials([[$class: 'FileBinding', credentialsId: 'PUF_FILE', variable: 'PUF_FILE']]) {
                    sh '''if [ "$REMOTE_BUILD_TOKEN" = "" ];then echo WONT WORK - REMOTE_BUILD_TOKEN not there;fi '''
                    sh '''if [ "$OSPC_API_KEY" = "" ];then echo WONT WORK - OSPC_API_KEY not there;fi '''
                    sh '''echo About to checkout scm'''
                    checkout scm
                    sh '''echo Checked out scm'''
                    sh '''cp $PUF_FILE Python/regression'''
                    sh '''cd Python/regression && bash run_reforms_bash.sh'''
                }
            }
        }
    }
}
