import { auth } from '@clerk/nextjs/server'
import { redirect } from 'next/navigation'
import VideoAnalyzer from '@/components/VideoAnalyzer'

export default async function DashboardPage() {
  const { userId } = await auth()
  
  if (!userId) {
    redirect('/sign-in')
  }

  return (
    <div className="container mx-auto py-8">
      <div className="text-center mb-8">
        <h1 className="text-3xl font-bold mb-2">
          Video Analysis Dashboard
        </h1>
        <p className="text-muted-foreground">
          Upload and analyze classroom videos for early autism indicators
        </p>
      </div>
      <VideoAnalyzer />
    </div>
  )
}
