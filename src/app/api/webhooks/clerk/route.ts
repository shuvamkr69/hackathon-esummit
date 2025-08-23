import { clerkClient } from "@clerk/nextjs/server";
import { NextResponse } from "next/server";
import { WebhookEvent } from "@clerk/nextjs/server";
import { headers } from "next/headers";
import { Webhook } from "svix";
import { createUserIfNotExists } from "@/lib/actions/user.action";

async function POST(request: Request) {
  const WEBHOOK_SECRET = process.env.WEBHOOK_SECRET;
  if (!WEBHOOK_SECRET) {
    return NextResponse.json(
      { error: "Webhook secret is not configured" },
      { status: 500 }
    );
  }

  const headerPayload = await headers();
  const svix_id = headerPayload.get("svix-id");
  const svix_timestamp = headerPayload.get("svix-timestamp");
  const svix_signature = headerPayload.get("svix-signature");

  if (!svix_id || !svix_timestamp || !svix_signature) {
    return NextResponse.json(
      { error: "Missing Svix headers" },
      { status: 400 }
    );
  }

  const payload = await request.json();
  const body = JSON.stringify(payload);

  const wh = new Webhook(WEBHOOK_SECRET);

  let evt: WebhookEvent;
  try {
    evt = wh.verify(body, {
      "svix-id": svix_id,
      "svix-timestamp": svix_timestamp,
      "svix-signature": svix_signature,
    }) as WebhookEvent;
  } catch (err) {
    console.error("Error verifying webhook:", err);
    return NextResponse.json(
      { error: "Invalid webhook signature" },
      { status: 400 }
    );
  }

  const { id } = evt.data;
  const eventType = evt.type;

  if (eventType === "user.created") {
    const { email_addresses, first_name, image_url } = evt.data;

    const user = {
      clerkId: id || "",
      email: email_addresses[0]?.email_address || "",
      name: first_name || "",
      avatarUrl: image_url || "",
    };
    console.log("New user created:", user);
    const newUser = await createUserIfNotExists(
      user.clerkId,
      user.email,
      user.name,
      user.avatarUrl
    );

    if (newUser && id) {
        await (await clerkClient()).users.updateUserMetadata(id, {
            publicMetadata: {
                userId : newUser,
            },
        });
    }
    console.log(`Webhook id  ${id} of type  ${eventType} received`);
    console.log("Webhook body:", body);

    return new Response("Webhook received", { status: 200 });
  }
}
