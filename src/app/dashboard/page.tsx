import { auth } from '@clerk/nextjs/server'
import { redirect } from 'next/navigation'
import Dashboard from '@/components/Dashboard'
import VideoAnalyzer from '@/components/VideoAnalyzer'

export default async function DashboardPage() {
  const { userId } = await auth()
  
  if (!userId) {
    redirect('/sign-in')
  }

  return (
    <>
      <VideoAnalyzer/>
    </>
  )
}
