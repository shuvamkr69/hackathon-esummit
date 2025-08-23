'use client';

import React, { useState } from 'react';
import { UserButton, useUser } from '@clerk/nextjs';
import { Brain } from 'lucide-react';
import Link from 'next/link';
import { 
  Navbar as ResizableNavbar, 
  NavBody, 
  NavItems, 
  MobileNav, 
  MobileNavHeader, 
  MobileNavMenu, 
  MobileNavToggle,
  NavbarButton 
} from '@/components/ui/resizable-navbar';

export default function Navbar() {
  const { isSignedIn } = useUser();
  const [isMenuOpen, setIsMenuOpen] = useState(false);

  const navItems = [
    { name: 'Home', link: '/' },
    { name: 'About', link: '/about' },
    ...(isSignedIn ? [{ name: 'Dashboard', link: '/dashboard' }] : []),
  ];

  const mobileNavItems = [
    { name: 'Home', link: '/' },
    { name: 'About', link: '/about' },
    ...(isSignedIn ? [{ name: 'Dashboard', link: '/dashboard' }] : []),
  ];

  return (
    <ResizableNavbar className="fixed top-0 left-0 right-0 z-50">
      {/* Desktop Navigation */}
      <NavBody className="bg-background/80 backdrop-blur-lg border-b border-border">
        {/* Logo */}
        <Link href="/" className="flex items-center space-x-2 relative z-20">
          <Brain className="h-8 w-8 text-primary" />
          <span className="font-bold text-xl bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
            AutismAI
          </span>
        </Link>

        {/* Navigation Items */}
        <NavItems items={navItems} />

        {/* Auth Buttons */}
        <div className="flex items-center space-x-4 relative z-20">
          {isSignedIn ? (
            <UserButton afterSignOutUrl="/" />
          ) : (
            <div className="flex items-center space-x-2">
              <NavbarButton 
                href="/sign-in" 
                variant="secondary"
                as={Link}
              >
                Sign In
              </NavbarButton>
              <NavbarButton 
                href="/sign-up" 
                variant="gradient"
                as={Link}
              >
                Get Started
              </NavbarButton>
            </div>
          )}
        </div>
      </NavBody>

      {/* Mobile Navigation */}
      <MobileNav className="bg-background/80 backdrop-blur-lg border-b border-border">
        <MobileNavHeader>
          {/* Mobile Logo */}
          <Link href="/" className="flex items-center space-x-2">
            <Brain className="h-7 w-7 text-primary" />
            <span className="font-bold text-lg bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
              AutismAI
            </span>
          </Link>

          {/* Mobile Auth & Toggle */}
          <div className="flex items-center space-x-3">
            {isSignedIn && (
              <UserButton afterSignOutUrl="/" />
            )}
            <MobileNavToggle 
              isOpen={isMenuOpen} 
              onClick={() => setIsMenuOpen(!isMenuOpen)} 
            />
          </div>
        </MobileNavHeader>

        {/* Mobile Menu */}
        <MobileNavMenu 
          isOpen={isMenuOpen} 
          onClose={() => setIsMenuOpen(false)}
          className="bg-background/95 backdrop-blur-lg border border-border"
        >
          <div className="flex flex-col space-y-4 w-full">
            {mobileNavItems.map((item, index) => (
              <Link
                key={index}
                href={item.link}
                onClick={() => setIsMenuOpen(false)}
                className="text-foreground hover:text-primary transition-colors py-2 px-4 rounded-md hover:bg-muted"
              >
                {item.name}
              </Link>
            ))}
            
            {!isSignedIn && (
              <div className="flex flex-col space-y-2 pt-4 border-t border-border">
                <Link
                  href="/sign-in"
                  onClick={() => setIsMenuOpen(false)}
                  className="text-center py-2 px-4 rounded-md border border-border hover:bg-muted transition-colors"
                >
                  Sign In
                </Link>
                <Link
                  href="/sign-up"
                  onClick={() => setIsMenuOpen(false)}
                  className="text-center py-2 px-4 rounded-md bg-gradient-to-r from-blue-600 to-purple-600 text-white hover:opacity-90 transition-opacity"
                >
                  Get Started
                </Link>
              </div>
            )}
          </div>
        </MobileNavMenu>
      </MobileNav>
    </ResizableNavbar>
  );
}
