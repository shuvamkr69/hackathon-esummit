"use server"

import User from "@/lib/models/user.model";
import {connectToDatabase} from "@/lib/db";

export async function createUserIfNotExists(clerkId: string, email: string, name?: string, avatarUrl?: string) {

    try {
        await connectToDatabase();
    
        let user = await User.findOne({clerkId});
    
        if (!user) {
            user = await User.create({clerkId, email, name, avatarUrl});
            return JSON.stringify(user);
        }
    } catch (error) {
        console.error("Error in createUserIfNotExists:", error);
        throw error;
    }


}