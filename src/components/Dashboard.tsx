'use client';

import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, LineChart, Line } from 'recharts';

const recentAnalyses = [
  { id: 'Analysis 1', status: 'Completed', date: '2024-01-15' },
  { id: 'Analysis 2', status: 'In Progress', date: '2024-01-16' },
  { id: 'Analysis 3', status: 'Completed', date: '2024-01-17' },
  { id: 'Analysis 4', status: 'Completed', date: '2024-01-18' },
  { id: 'Analysis 5', status: 'Completed', date: '2024-01-19' },
];

const eyeContactData = [
  { day: 'Mon', value: 40 },
  { day: 'Tue', value: 60 },
  { day: 'Wed', value: 85 },
  { day: 'Thu', value: 50 },
  { day: 'Fri', value: 30 },
  { day: 'Sat', value: 45 },
  { day: 'Sun', value: 55 },
];

const gesturesData = [
  { day: 'Mon', value: 70 },
  { day: 'Tue', value: 60 },
  { day: 'Wed', value: 75 },
  { day: 'Thu', value: 50 },
  { day: 'Fri', value: 65 },
  { day: 'Sat', value: 80 },
  { day: 'Sun', value: 85 },
];

export default function Dashboard() {
  return (
    <div className="max-w-6xl mx-auto p-6 space-y-8">
      <h1 className="text-2xl font-bold">Dashboard</h1>

      {/* Overview Section */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <Card className='transition delay-50 duration-150 ease-in-out hover:-translate-y-1 hover:scale-105 border-2 hover:border-primary/50 transition-colors'>
          <CardHeader>
            <CardTitle>Risk Score</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-xl font-semibold text-gray-400">Low</p>
          </CardContent>
        </Card>

        <Card className='transition delay-50 duration-150 ease-in-out hover:-translate-y-1 hover:scale-105 border-2 hover:border-primary/50 transition-colors'>
          <CardHeader>
            <CardTitle>Key Indicators Detected</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-lg font-medium text-gray-400">Eye Contact, Gestures, Interactions</p>
          </CardContent>
        </Card>

        <Card className='transition delay-50 duration-150 ease-in-out hover:-translate-y-1 hover:scale-105 border-2 hover:border-primary/50 transition-colors'>
          <CardHeader>
            <CardTitle>Number of Analyses Run</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-xl font-semibold text-gray-400">12</p>
          </CardContent>
        </Card>
      </div>

      {/* Recent Analyses */}
      <div>
        <h2 className="text-lg font-semibold mb-4">Recent Analyses</h2>
        <Card>
          <CardContent>
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Analysis ID</TableHead>
                  <TableHead>Status</TableHead>
                  <TableHead>Date</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {recentAnalyses.map((analysis, idx) => (
                  <TableRow key={idx}>
                    <TableCell>{analysis.id}</TableCell>
                    <TableCell>
                      <span
                        className={`px-3 py-1 rounded text-sm ${
                          analysis.status === 'Completed'
                            ? 'bg-green-100 text-green-700'
                            : 'bg-blue-100 text-blue-700'
                        }`}
                      >
                        {analysis.status}
                      </span>
                    </TableCell>
                    <TableCell>{analysis.date}</TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </CardContent>
        </Card>
      </div>

      {/* Behavioral Trends */}
      <div>
        <h2 className="text-lg font-semibold mb-4">Behavioral Trends</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* Eye Contact */}
          <Card className='transition delay-50 duration-150 ease-in-out hover:-translate-y-1 hover:scale-105 border-2 hover:border-primary/50 transition-colors'>
            <CardHeader>
              <CardTitle>Eye Contact</CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-2xl font-bold">80%</p>
              <p className="text-sm text-green-600 mb-4">Last 7 Days +5%</p>
              <div className="h-48">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={eyeContactData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="day" />
                    <YAxis />
                    <Tooltip />
                    <Bar dataKey="value" fill="#3b82f6" />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </CardContent>
          </Card>

          {/* Gestures */}
          <Card className='transition delay-50 duration-150 ease-in-out hover:-translate-y-1 hover:scale-105 border-2 hover:border-primary/50 transition-colors'>
            <CardHeader>
              <CardTitle>Gestures</CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-2xl font-bold">65%</p>
              <p className="text-sm text-red-600 mb-4">Last 7 Days -2%</p>
              <div className="h-48">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={gesturesData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="day" />
                    <YAxis />
                    <Tooltip />
                    <Line type="monotone" dataKey="value" stroke="#3b82f6" strokeWidth={2} />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
}
