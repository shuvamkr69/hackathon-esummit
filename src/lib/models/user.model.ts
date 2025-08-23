import {Schema, model, models} from 'mongoose';

const UserSchema = new Schema({
    clerkId: {type: String, required: true, unique: true},
    email: {type: String, required: true, unique: true},
    name: {type: String},
    avatarUrl: {type: String},
    role: {type: String, enum: ['user', 'admin'], default: 'user'},
}, {timestamps: true});

const User = models.User || model('User', UserSchema);

export default User;