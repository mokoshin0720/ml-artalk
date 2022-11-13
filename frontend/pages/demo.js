import Image from 'next/image'

export default function Demo() {
    return (
        <div>
            <Image src="/adriaen-brouwer_feeling.jpg" alt="Vercel Logo" layout='fill' objectFit='contain' />
        </div>
    )
}