import MainHero from '@/components/MainHero';
import KeyFeatures from '@/components/KeyFeatures';
import HowItWorks from '@/components/HowItWorks';
import CTASection from '@/components/CTASection';

export default function Home() {
  return (
    <main>
      <MainHero />
      <KeyFeatures />
      <HowItWorks />
      <CTASection />
    </main>
  );
}
