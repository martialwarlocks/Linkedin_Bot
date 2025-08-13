DEPLOYED.

Service URL: https://sales-assistant-service-822359826336.asia-south1.run.app 

Run to Update:

gcloud builds submit --tag asia-south1-docker.pkg.dev/spikedai/my-app-repo/sales-assistant-app:latest

gcloud run deploy sales-assistant-service --image asia-south1-docker.pkg.dev/spikedai/my-app-repo/sales-assistant-app:latest --platform managed --region asia-south1 --allow-unauthenticated --set-secrets="GROQ_API_KEY=groq-api-key:latest" --update-env-vars="GCS_BUCKET_NAME=spikedai-bucket-one" --memory 4Gi --cpu 2 --timeout 600s
