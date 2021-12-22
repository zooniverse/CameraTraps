#!groovy

pipeline {
  agent none

  options {
    disableConcurrentBuilds()
  }

  stages {
    stage('Build Docker image') {
      agent any
      steps {
        script {
          def dockerRepoName = 'zooniverse/camera-traps-api'
          def dockerImageName = "${dockerRepoName}:${GIT_COMMIT}"
          def newImage = docker.build(dockerImageName)
          newImage.push()

          if (BRANCH_NAME == 'zooniverse-deployment') {
            stage('Update latest tag') {
              newImage.push('latest')
            }
          }
        }
      }
    }
    stage('Dry run deployments') {
      agent any
      steps {
        sh "sed 's/__IMAGE_TAG__/${GIT_COMMIT}/g' kubernetes/deployment.tmpl | kubectl --context azure apply --dry-run=client --record -f -"
      }
    }

    stage('Deploy to Kubernetes') {
      when { branch 'zooniverse-deployment' }
      agent any
      steps {
        sh "sed 's/__IMAGE_TAG__/${GIT_COMMIT}/g' kubernetes/deployment.tmpl | kubectl --context azure apply --record -f -"
      }
    }
  }
}
