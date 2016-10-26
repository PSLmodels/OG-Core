node {
    withCredentials([[$class: 'StringBinding', credentialsId: 'OSPC_API_KEY', variable: 'OSPC_API_KEY']]) {
        sh '''if [ "$OSPC_API_KEY" = "" ];then echo WONT WORK;fi '''
        sh '''echo About to checkout scm'''
        checkout scm
        sh '''echo Checked out scm'''
        sh "export OSPC_API_KEY=$OSPC_API_KEY && env | sort && echo pwd is `pwd` && cd Python/regression && bash run_reforms_bash.sh"
    }
}
