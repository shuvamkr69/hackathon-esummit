import mongoose, {Mongoose} from 'mongoose';

const MONGO_URI = process.env.MONGO_URI;

interface MongooseConnection {
    conn: Mongoose | null;
    promise: Promise<Mongoose> | null;
}

// Extend the global object to include mongoose
declare global {
    var mongoose: MongooseConnection | undefined;
}

let cached: MongooseConnection = global.mongoose || { conn: null, promise: null };

if (!cached) {
    cached = global.mongoose = { conn: null, promise: null };
}

export async function connectToDatabase() {
    if (cached.conn) {
        return cached.conn;
    }

    if (!MONGO_URI) {
        throw new Error('Please define the MONGO_URI environment variable inside .env.local');
    }

    cached.promise = cached.promise ||
     mongoose.connect(MONGO_URI, 
    {
        bufferCommands: false,
        connectTimeoutMS: 60000,
    });
    cached.conn = await cached.promise;
    return cached.conn;
}
